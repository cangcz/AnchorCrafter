import inspect
import math
import os.path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import einops
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion \
    import _resize_with_antialiasing, _append_dims
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from anchor_crafter.modules.track_net import TrackNet
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import constants

from ..modules.obj_proj_net import ObjProjNet
from ..modules.obj_attn_net import ObjAttnNet
from ..modules.pose_hand_net import PoseNet


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")

    return outputs


@dataclass
class MimicMotionPipelineOutput(BaseOutput):
    r"""
    Output class for mimicmotion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class AnchorCrafterPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K]
            (https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
        pose_net ([`PoseNet`]):
            A `` to inject pose signals into unet.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
            self,
            vae: AutoencoderKLTemporalDecoder,
            image_encoder: CLIPVisionModelWithProjection,
            obj_image_encoder: AutoModel,
            unet: UNetSpatioTemporalConditionModel,
            scheduler: EulerDiscreteScheduler,
            feature_extractor: CLIPImageProcessor,
            dino_feature_extractor: AutoImageProcessor,
            pose_net: PoseNet,
            track_net: TrackNet,
            obj_proj_net: ObjProjNet,
            obj_attn_net: ObjAttnNet
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            obj_image_encoder=obj_image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            dino_feature_extractor=dino_feature_extractor,
            pose_net=pose_net,
            track_net=track_net,
            obj_proj_net=obj_proj_net,
            obj_attn_net=obj_attn_net
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
            self,
            image: PipelineImageInput,
            obj_pixels: PipelineImageInput,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool):
        dtype = next(self.image_encoder.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds

        obj_all_embeddings = None
        for obj in obj_pixels:
            if not isinstance(obj, torch.Tensor):
                obj = self.image_processor.pil_to_numpy(obj)
                obj = self.image_processor.numpy_to_pt(obj)

                # We normalize the image before resizing to match with the original implementation.
                # Then we unnormalize it after resizing.
                obj = obj * 2.0 - 1.0
                obj = _resize_with_antialiasing(obj, (518, 518))
                obj = (obj + 1.0) / 2.0

                # Normalize the image with for CLIP input
                obj = self.dino_feature_extractor(
                    images=obj,
                    do_normalize=True,
                    do_center_crop=False,
                    do_resize=False,
                    do_rescale=False,
                    return_tensors="pt",
                ).pixel_values

            obj = obj.to(device=device, dtype=dtype)

            obj_pixels_embeddings = self.obj_image_encoder(obj).last_hidden_state

            if obj_all_embeddings is None:
                obj_all_embeddings = obj_pixels_embeddings
            else:
                obj_all_embeddings = torch.concat((obj_all_embeddings, obj_pixels_embeddings), dim=1)
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        return image_embeddings, obj_all_embeddings

    def _encode_vae_image(
            self,
            image: torch.Tensor,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device, dtype=self.vae.dtype)

        image_latents = torch.zeros((image.shape[0], 4, image.shape[-2]//8, image.shape[-1]//8)).to(device=device, dtype=self.vae.dtype)
        for i in range(0, image.shape[0], 16):
            if i + 16 > image.shape[0]:
                image_latents[i:] = self.vae.encode(image[i:]).latent_dist.mode()
            else:
                image_latents[i:i + 16] = self.vae.encode(image[i:i + 16]).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
            self,
            fps: int,
            motion_bucket_id: int,
            noise_aug_strength: float,
            dtype: torch.dtype,
            batch_size: int,
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, " \
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. " \
                f"Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(
            self,
            latents: torch.Tensor,
            num_frames: int,
            decode_chunk_size: int = 8):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame.cpu())
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
            self,
            batch_size: int,
            num_frames: int,
            num_channels_noise_latents: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            generator: torch.Generator,
            latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_noise_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
            self,
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],  # image_pixels
            image_pose: Union[torch.FloatTensor],  # pose_pixels
            obj_pixels: Union[torch.FloatTensor],  # obj_pixels
            obj_track_pixels:  Union[torch.FloatTensor],  # obj_track_pixels
            hand_pixels: Union[torch.FloatTensor],  # hand_pixels
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            tile_size: Optional[int] = 16,
            tile_overlap: Optional[int] = 4,
            num_inference_steps: int = 25,
            min_guidance_scale: float = 1.0,
            max_guidance_scale: float = 3.0,
            fps: int = 7,
            motion_bucket_id: int = 127,
            noise_aug_strength: float = 0.02,
            image_only_indicator: bool = False,
            decode_chunk_size: Optional[int] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            return_dict: bool = True,
            device: Union[str, torch.device] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/
                feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` 
                and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second.The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. 
                The higher the number the more motion will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, 
                the higher it is the less the video will look like the init image. Increase it for more motion.
            image_only_indicator (`bool`, *optional*, defaults to False):
                Whether to treat the inputs as batch of images instead of videos.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time.The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. 
                By default, the decoder will decode all frames at once for maximal quality. 
                Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            device:
                On which device the pipeline runs on.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, 
                [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image(
        "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        pose_pixels = torch.cat([image_pose, hand_pixels], dim=1)
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = device if device is not None else self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        self.image_encoder.to(device)
        self.obj_image_encoder.to(device)
        encoder_hidden_states, obj_embeddings = self._encode_image(image, obj_pixels, device, num_videos_per_prompt,
                                              self.do_classifier_free_guidance)

        self.image_encoder.cpu()
        self.obj_image_encoder.cpu()
        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        obj_image = pil_to_tensor(obj_pixels[1])
        h_pad = (image.shape[-2] - obj_image.shape[-2]) // 2
        w_pad = (image.shape[-1] - obj_image.shape[-1]) // 2
        obj_image = F.pad(obj_image, (w_pad, w_pad, h_pad, h_pad), mode='constant', value=0)
        obj_image = self.image_processor.preprocess(obj_image, height=height, width=width).to(device)
        
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise
        obj_image = obj_image + noise_aug_strength * noise

        self.vae.to(device)
        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(encoder_hidden_states.dtype)
        obj_image_latents = self._encode_vae_image(
            obj_image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        obj_image_latents = obj_image_latents.to(encoder_hidden_states.dtype)
        self.vae.cpu()

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        obj_image_latents = obj_image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            encoder_hidden_states.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            tile_size,
            4,
            height,
            width,
            encoder_hidden_states.dtype,
            device,
            generator,
            latents,
        )
        latents = latents.repeat(1, num_frames // tile_size + 1, 1, 1, 1)[:, :num_frames]
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        self._num_timesteps = len(timesteps)

        self.pose_net.to(device)
        self.track_net.to(device)
        self.unet.to(device)
        self.obj_proj_net.to(device)
        self.obj_attn_net.to(device)

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        obj_cls_emb = torch.cat([
            obj_embeddings[:, 0, :], obj_embeddings[:, 1370, :], obj_embeddings[:, 1370*2, :]
        ], dim=1)
        obj_cls_embeddings = self.obj_proj_net(obj_cls_emb)
        obj_attn_embeddings = self.obj_attn_net(obj_embeddings)
        encoder_hidden_states = torch.concat([
            encoder_hidden_states, obj_cls_embeddings, obj_attn_embeddings
        ], dim=1)

        if self.do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(encoder_hidden_states)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            encoder_hidden_states = torch.cat([negative_image_embeddings, encoder_hidden_states])

        bias_start = 1
        bias_step = 4
        with (self.progress_bar(total=len(timesteps) * math.ceil((num_frames-1)/(tile_size-1))) as progress_bar):
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents, obj_image_latents], dim=2)

                # predict the noise residual
                noise_pred = torch.zeros_like(image_latents)
                noise_pred_cnt = torch.zeros_like(image_latents)
                weight = torch.ones_like(image_latents)

                bias_start = (bias_start - 1) % (num_frames - 1) + 1
                start_cur = bias_start
                finished_len = 1
                while finished_len < num_frames:
                    start_cur = (start_cur - 1) % (num_frames - 1) + 1
                    end_cur = start_cur + tile_size - 1

                    idx = [0, ]
                    idx.extend([(ii - 1) % (num_frames - 1) + 1 for ii in range(start_cur, end_cur)])

                    # classification-free inference
                    pose_latents = self.pose_net(pose_pixels[idx].to(device))

                    track_latents = self.track_net(obj_track_pixels[idx].to(device))


                    _noise_pred = self.unet(
                        latent_model_input[:1, idx],
                        t,
                        encoder_hidden_states=encoder_hidden_states[:1],
                        added_time_ids=added_time_ids[:1],
                        pose_latents=None,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                        obj_track_latents=None,
                    )[0]
                    noise_pred[:1, idx] += _noise_pred

                    # normal inference

                    _noise_pred = self.unet(
                        latent_model_input[1:, idx],
                        t,
                        encoder_hidden_states=encoder_hidden_states[1:],
                        added_time_ids=added_time_ids[1:],
                        pose_latents=pose_latents,
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                        obj_track_latents= track_latents,
                    )[0]
                    noise_pred[1:, idx] += _noise_pred

                    noise_pred_cnt[:, idx] += weight[:, idx]
                    finished_len += tile_size - 1
                    start_cur += tile_size - 1
                    progress_bar.update()

                bias_start += bias_step
                noise_pred = noise_pred.div_(noise_pred_cnt)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

        self.pose_net.cpu()
        self.unet.cpu()
        self.track_net.cpu()
        self.obj_proj_net.cpu()
        if not output_type == "latent":
            self.vae.decoder.to(device)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return MimicMotionPipelineOutput(frames=frames)
