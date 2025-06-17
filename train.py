import argparse
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from einops import rearrange

import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from anchor_crafter.modules.attention_processor import IPAttnProcessor
from diffusers.models.attention_processor import XFormersAttnProcessor

from typing import Union
from torchvision.transforms.functional import to_pil_image

import copy

from anchor_crafter.modules.unet import UNetSpatioTemporalConditionModel
from anchor_dataset import AnchorDataset
from anchor_crafter.utils.loader import AnchorCrafterModel

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

import multiprocessing

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, unet, pose_net, track_net, obj_proj_net, obj_attn_net):
        super().__init__()
        self.unet = unet  # U-Net model for denoising and image generation
        self.pose_net = pose_net  # Pose network for processing human posture features
        self.track_net = track_net  # Object tracking network for tracking target movements
        self.obj_proj_net = obj_proj_net  # Object feature projection network
        self.obj_attn_net = obj_attn_net  # Object attention network

    def forward(self, pose_features, inp_noisy_latents, conditional_latents,
                timesteps, encoder_hidden_states, added_time_ids,
                track_features, conditional_latents_obj, hand_features,
                obj_embeddings, conditioning_dropout_prob=None, generator=None):

        # Concatenate hand features with pose features
        pose_features = torch.cat([pose_features, hand_features], dim=2)
        pose_features = self.pose_net(pose_features)  # Process pose features
        track_features = self.track_net(track_features)  # Process object tracking features

        # Extract object class embeddings
        cls_indices = [0, 1370, 1370 * 2]  # Class indices for object categories
        obj_cls_emb = torch.cat([obj_embeddings[:, idx, :] for idx in cls_indices], dim=1)
        obj_cls_embeddings = self.obj_proj_net(obj_cls_emb)  # Project object class embeddings
        obj_attn_embeddings = self.obj_attn_net(obj_embeddings)  # Apply attention to object embeddings

        # Combine encoder hidden states with object features
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        encoder_hidden_states = torch.cat([
            encoder_hidden_states, obj_cls_embeddings, obj_attn_embeddings
        ], dim=1)

        # Apply conditioning dropout (used for classifier-free guidance)
        if conditioning_dropout_prob is not None:
            bsz = inp_noisy_latents.shape[0]
            random_p = torch.rand(bsz, device=inp_noisy_latents.device, generator=generator)

            # Sample mask to randomly drop edit prompts
            prompt_mask = random_p < 2 * conditioning_dropout_prob
            # image_mask = 1 - ((random_p >= conditioning_dropout_prob) * (random_p < 3 * conditioning_dropout_prob))
            image_mask_dtype = conditional_latents.dtype
            image_mask = 1 - ( (random_p >= conditioning_dropout_prob).to(image_mask_dtype) * (random_p < 3 * conditioning_dropout_prob).to(image_mask_dtype))
            # Apply mask to remove parts of encoder hidden states
            encoder_hidden_states = torch.where(
                prompt_mask.reshape(bsz, 1, 1), torch.zeros_like(encoder_hidden_states), encoder_hidden_states
            )

            # Apply dropout to image latent conditioning
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            conditional_latents *= image_mask
            conditional_latents_obj *= image_mask
            pose_features = pose_features if image_mask[0, 0, 0, 0] != 0 else None
            track_features = track_features if image_mask[0, 0, 0, 0] != 0 else None

        # Expand and concatenate conditional latents
        conditional_latents = conditional_latents.unsqueeze(1).expand(-1, inp_noisy_latents.shape[1], -1, -1, -1)
        conditional_latents_obj = conditional_latents_obj.unsqueeze(1).expand(-1, inp_noisy_latents.shape[1], -1, -1, -1)
        inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents, conditional_latents_obj], dim=2)

        # Pass through U-Net for denoising and final prediction
        model_pred = self.unet(
            inp_noisy_latents, timesteps, encoder_hidden_states, pose_latents=pose_features,
            obj_track_latents=track_features, added_time_ids=added_time_ids
        ).sample
        return model_pred

# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
                      dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1).float()

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return (gauss / gauss.sum(-1, keepdim=True)).half()


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Anchor Crafter."
    )
    parser.add_argument(
        "--base_folder",
        required=True,
        type=str,
        help="Path to HOI dataset.",
    )
    parser.add_argument(
        "--noobj_folder",
        required=True,
        type=str,
        help="Path to no object dataset.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dino_path",
        type=str,
        default=None,
        required=True,
        help="Path to dino model.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=576,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5, 
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None
    )
    parser.add_argument(
        '--loss_rate_hoi',
        type=float,
        default=1
    )
    parser.add_argument(
        '--finetune',
        action='store_true'
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def main():
    multiprocessing.set_start_method("spawn")

    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed,True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    anchorcrafter_models = AnchorCrafterModel(args.pretrained_model_name_or_path, args.dino_path)

    # 在设置Adapter前设置xformer
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            anchorcrafter_models.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # 设置Adapter
    attn_procs = {}
    for name in anchorcrafter_models.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith(
            "attn1.processor") else anchorcrafter_models.unet.config.cross_attention_dim
        hidden_size = None
        if name.startswith("mid_block"):
            hidden_size = anchorcrafter_models.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(anchorcrafter_models.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = anchorcrafter_models.unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = XFormersAttnProcessor()
        else:
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=19
            )
    anchorcrafter_models.unet.set_attn_processor(attn_procs)

    anchor_weight = torch.load(args.ckpt_path, map_location="cpu")
    if 'module' in anchor_weight:    # load from checkpoint
        anchor_weight = anchor_weight['module']

    missing_keys, unexpected_keys = anchorcrafter_models.load_state_dict(anchor_weight, strict=False)
    print("missing_keys", len(missing_keys))
    print("unexpected_keys", len(unexpected_keys))
    print("missing_keys: 1333  unexpected_keys: 0 is right")

    vae = anchorcrafter_models.vae
    image_encoder = anchorcrafter_models.image_encoder
    obj_image_encoder = anchorcrafter_models.obj_image_encoder
    unet = anchorcrafter_models.unet
    feature_extractor = anchorcrafter_models.feature_extractor
    dino_feature_extractor = anchorcrafter_models.dino_feature_extractor
    pose_net = anchorcrafter_models.pose_net
    track_net = anchorcrafter_models.track_net
    obj_proj_net = anchorcrafter_models.obj_proj_net
    obj_attn_net = anchorcrafter_models.obj_attn_net
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    obj_image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(True)
    track_net.requires_grad_(True)
    obj_proj_net.requires_grad_(True)
    obj_attn_net.requires_grad_(True)
    net = Net(unet, pose_net, track_net, obj_proj_net=obj_proj_net, obj_attn_net=obj_attn_net)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    obj_image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps *
                args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    parameters_list = []
    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.
    for name, para in net.named_parameters():
        if 'unet' in name:
           parameters_list.append(para)
           para.requires_grad = True
           para.data = para.data.to(dtype=torch.float32)
        else:
            para.requires_grad = False
            para.data = para.data.to(dtype=weight_dtype)  # torch.float16
    for name, param in net.track_net.named_parameters():
        parameters_list.append(param)
        param.requires_grad = True
        param.data = param.data.to(dtype=torch.float32)
    for name, param in net.obj_proj_net.named_parameters():
        parameters_list.append(param)
        param.requires_grad = True
        param.data = param.data.to(dtype=torch.float32)
    for name, param in net.obj_attn_net.named_parameters():
        parameters_list.append(param)
        param.requires_grad = True
        param.data = param.data.to(dtype=torch.float32)
    for name, param in net.pose_net.named_parameters():
        parameters_list.append(param)
        param.requires_grad = True
        param.data = param.data.to(dtype=torch.float32)


    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open('params_freeze.txt', 'w')
        rec_txt2 = open('params_train.txt', 'w')
        for name, para in net.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = AnchorDataset(args.base_folder, args.noobj_folder, resolution=args.resolution, sample_frames=args.num_frames)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    net, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        net, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # attribute handling for models using DDP
    if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        net = net.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!

    total_batch_size = args.per_gpu_batch_size * \
                       accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def _encode_image(
            image: Image,
            obj_pixels: Image,
            device: Union[str, torch.device],
    ):

        if not isinstance(image, torch.Tensor):
            image = image_processor.pil_to_numpy(image)
            image = image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=weight_dtype)
        image_embeddings = image_encoder(image).image_embeds

        obj_all_embeddings = []
        for obj in obj_pixels:
            if not isinstance(obj, torch.Tensor):
                obj = image_processor.pil_to_numpy(obj)
                obj = image_processor.numpy_to_pt(obj)

                # We normalize the image before resizing to match with the original implementation.
                # Then we unnormalize it after resizing.
                obj = obj * 2.0 - 1.0
                obj = _resize_with_antialiasing(obj, (518, 518))
                obj = (obj + 1.0) / 2.0

                # Normalize the image with for dino input
                obj = dino_feature_extractor(
                    images=obj,
                    do_normalize=True,
                    do_center_crop=False,
                    do_resize=False,
                    do_rescale=False,
                    return_tensors="pt",
                ).pixel_values

            obj = obj.to(device=device, dtype=weight_dtype)

            obj_embeddings = obj_image_encoder(obj).last_hidden_state

            obj_all_embeddings.append(obj_embeddings)
        # Concatenate all object embeddings along dimension 1
        obj_all_embeddings = torch.cat(obj_all_embeddings, dim=1) if obj_all_embeddings else None

        return image_embeddings, obj_all_embeddings

    def _get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            dtype,
            batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = net.unet.config.addition_time_embed_dim * \
                               len(add_time_ids)
        expected_add_embed_dim = net.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                    num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        net.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with ((accelerator.accumulate(net))):
                pose_pixels = batch["pose_pixels"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                image_pixels = batch["image_pixels"].to(weight_dtype).to(  # people_pixels
                    accelerator.device, non_blocking=True
                )
                video_pixels = batch["video_pixels"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                obj_pixels = batch["obj_pixels"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                box_video_pixels = batch["box_video_pixels"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                hand_pixels = batch["hand_pixels"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                image_condition = image_pixels
                obj_condition = obj_pixels[:, 1:2]

                image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels.squeeze(1) + 1.0) * 127.5]
                obj_pixels = [to_pil_image(img.to(torch.uint8)) for img in (obj_pixels.squeeze(0) + 1.0) * 127.5]

                encoder_hidden_states, obj_embeddings = _encode_image(image_pixels, obj_pixels, accelerator.device)

                conditional_pixel_values = image_condition
                conditional_pixel_values_obj = obj_condition

                h_pad = (conditional_pixel_values.shape[-2] - conditional_pixel_values_obj.shape[-2]) // 2
                w_pad = (conditional_pixel_values.shape[-1] - conditional_pixel_values_obj.shape[-1]) // 2
                conditional_pixel_values_obj = F.pad(conditional_pixel_values_obj, (w_pad, w_pad, h_pad, h_pad), mode='constant', value=0)

                latents = tensor_to_vae_latent(video_pixels, vae)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                cond_sigmas = rand_log_normal(shape=[bsz, ], loc=-3.0, scale=0.5).to(latents)
                noise_aug_strength = cond_sigmas[0]  # TODO: support batch > 1
                cond_sigmas = cond_sigmas[:, None, None, None, None]
                conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                conditional_pixel_values_obj = \
                    torch.randn_like(conditional_pixel_values_obj) * cond_sigmas + conditional_pixel_values_obj
                conditional_latents_obj = tensor_to_vae_latent(conditional_pixel_values_obj, vae)[:, 0, :, :, :]
                conditional_latents_obj = conditional_latents_obj / vae.config.scaling_factor

                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz, ], loc=0.7, scale=1.6).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)

                inp_noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    7,  # fixed
                    127,  # motion_bucket_id = 127, fixed
                    noise_aug_strength,  # noise_aug_strength == cond_sigmas
                    encoder_hidden_states.dtype,
                    bsz,
                )
                added_time_ids = added_time_ids.to(latents.device)

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = latents

                model_pred = net(
                    pose_pixels,
                    inp_noisy_latents,
                    conditional_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_time_ids,
                    box_video_pixels,
                    conditional_latents_obj=conditional_latents_obj,
                    hand_features=hand_pixels,
                    obj_embeddings=obj_embeddings,
                    conditioning_dropout_prob=args.conditioning_dropout_prob,
                    generator=generator
                )
                box_video_pixels = box_video_pixels[:, :, 1, :, :]
                # Denoise the latents
                c_out = -sigmas / ((sigmas ** 2 + 1) ** 0.5)
                c_skip = 1 / (sigmas ** 2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents  # torch.Size([1, 17, 4, 128, 64])
                weighing = (1 + sigmas ** 2) * (sigmas ** -2.0)

                loss_weight = []
                ori_H = box_video_pixels.shape[2]  # 1024
                tar_H = denoised_latents.shape[3]  # 128
                ratio = ori_H // tar_H
                total_area = video_pixels.shape[-1] * video_pixels.shape[-2] / (ratio * ratio)

                with torch.no_grad():
                    hand_pixels_sum = hand_pixels.sum(dim=2)
                    for idx in range(box_video_pixels.shape[1]):
                        no_zero = torch.zeros_like(box_video_pixels[0][idx])
                        no_zero[box_video_pixels[0][idx] != -1] = 1
                        if not args.finetune:
                            no_zero[hand_pixels_sum[0][idx] != -3] = 1
                        
                        no_zero = no_zero.unsqueeze(0).unsqueeze(0)
                        no_zero = nn.functional.interpolate(
                            no_zero, size=(denoised_latents.shape[-2], denoised_latents.shape[-1]), mode='nearest'
                        )
                        no_zero = no_zero.squeeze(0).squeeze(0).float()
                        part_area = no_zero.sum()
                        if part_area == 0:
                            loss_weight.append(
                                torch.ones((denoised_latents.shape[-2], denoised_latents.shape[-1]), dtype=torch.float32)
                            )
                            continue
                        resize_weight = (total_area / part_area) * args.loss_rate_hoi
                        loss_weight_frame = torch.ones(
                            (denoised_latents.shape[-2], denoised_latents.shape[-1]), dtype=torch.float32
                        )
                        loss_weight_frame[no_zero == 1] = resize_weight
                        loss_weight.append(loss_weight_frame)
                    loss_weight = torch.stack(loss_weight, dim=0)   # f, h, w
                    loss_weight = loss_weight.unsqueeze(1).unsqueeze(0)     # b, f, c, h, w
                    loss_weight = loss_weight.repeat((1, 1, 4, 1, 1)).cuda()

                # MSE loss
                loss = torch.mean(
                    torch.mul((weighing.float() * (denoised_latents.float() -
                                                target.float()) ** 2), loss_weight).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # save checkpoints!
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit and accelerator.is_main_process:
                            num_to_remove = len(
                                checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.wait_for_everyone()
                    print('start save state')
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
