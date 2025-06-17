import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..modules.unet import UNetSpatioTemporalConditionModel
import torch.nn as nn
from anchor_crafter.modules.track_net import TrackNet

from packaging import version
from diffusers.utils.import_utils import is_xformers_available

from anchor_crafter.modules.attention_processor import IPAttnProcessor
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import AutoImageProcessor, AutoModel

from ..pipelines.pipeline import AnchorCrafterPipeline
from ..modules.obj_proj_net import ObjProjNet
from ..modules.obj_attn_net import ObjAttnNet
from ..modules.pose_hand_net import PoseNet

logger = logging.getLogger(__name__)


class AnchorCrafterModel(torch.nn.Module):
    def __init__(self, base_model_path, dino_path):
        """construnct base model components and load pretrained svd model except pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"))

        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=torch.float16, variant="fp16")

        self.obj_image_encoder = AutoModel.from_pretrained(dino_path)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder", torch_dtype=torch.float16, variant="fp16")
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        self.dino_feature_extractor = AutoImageProcessor.from_pretrained(dino_path)
        # pose_net
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])
        # track_net
        self.track_net = TrackNet(noise_latent_channels=self.unet.config.block_out_channels[0])
        self.obj_proj_net = ObjProjNet(clip_extra_context_tokens=3)
        self.obj_attn_net = ObjAttnNet()


def create_pipeline(infer_config, device):
    """create mimicmotion pipeline and load pretrained weight

    Args:
        infer_config (str): 
        device (str or torch.device): "cpu" or "cuda:{device_id}"
    """
    anchorcrafter_models = AnchorCrafterModel(infer_config.base_model_path, infer_config.dino_path)

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

    # Adapter
    attn_procs = {}
    for name in anchorcrafter_models.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else anchorcrafter_models.unet.config.cross_attention_dim
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
    if infer_config.new_path[-2:] == "pt":
        # Load the state dict from a .pt file
        anchorcrafter_models.load_state_dict(torch.load(infer_config.new_path, map_location="cpu")["module"],
                                             strict=False)
    else:
        missing_keys, unexpected_keys = anchorcrafter_models.load_state_dict(torch.load(infer_config.new_path, map_location="cpu"),
                                                strict=False)
    print("missing_keys", len(missing_keys))
    print("unexpected_keys", len(unexpected_keys))

    pipeline = AnchorCrafterPipeline(
        vae=anchorcrafter_models.vae,
        image_encoder=anchorcrafter_models.image_encoder,
        obj_image_encoder=anchorcrafter_models.obj_image_encoder,
        unet=anchorcrafter_models.unet,
        scheduler=anchorcrafter_models.noise_scheduler,
        feature_extractor=anchorcrafter_models.feature_extractor,
        dino_feature_extractor=anchorcrafter_models.dino_feature_extractor,
        pose_net=anchorcrafter_models.pose_net,
        track_net=anchorcrafter_models.track_net,
        obj_proj_net=anchorcrafter_models.obj_proj_net,
        obj_attn_net=anchorcrafter_models.obj_attn_net
    )

    return pipeline
