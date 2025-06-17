import os
import argparse
import logging
import math

from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image

from anchor_crafter.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from constants import ASPECT_RATIO
import decord
from PIL import Image
import torch.nn.functional as F
import random
from anchor_crafter.utils.utils import save_to_mp4
from anchor_crafter.dwpose.preprocess import get_video_pose, get_image_pose, align_track_video
from anchor_crafter.pipelines.pipeline import AnchorCrafterPipeline
from anchor_crafter.utils.loader import create_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_video_reference(video_path, sample_stride, resolution, num_frames, p_scale, p_bias):
    """
    Reads a video using decord, resizes frames, limits the frame count, and aligns frames.
    
    Args:
        video_path (str): Path to the video.
        sample_stride (int): Frame sampling stride.
        resolution (int): Target resolution.
        num_frames (int): Number of frames to process (-1 means process all frames).
        p_scale (tuple): Scaling factors.
        p_bias (tuple): Bias parameters.
    
    Returns:
        np.ndarray: Aligned video frames with a zero frame inserted at the beginning.
    """
    # Read the video using decord
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    
    # Limit the number of frames if needed
    if num_frames != -1:
        frames = frames[:num_frames, ...]
    
    # Convert frames to a torch tensor with shape (F, C, H, W)
    frames = torch.from_numpy(frames).permute((0, 3, 1, 2))
    
    # Resize the frames to the target dimensions
    frames = Resize([int(resolution / ASPECT_RATIO), resolution])(frames)
    
    # Align the frames using the provided scale and bias parameters
    aligned_frames = align_track_video(frames, p_scale, p_bias)
    
    # Insert a zero frame at the beginning
    zero_frame = torch.zeros(aligned_frames[0].shape)
    aligned_frames = np.concatenate([zero_frame.unsqueeze(0), aligned_frames])
    
    return aligned_frames


def preprocess(video_path, image_path, obj_path, obj_track_path, hand_path,
               resolution=576, sample_stride=2, num_frames=-1):
    """
    Preprocess the reference image pose and video pose, and extract additional references for object appearance,
    object trajectory, and hand video.

    Args:
        video_path (str): Path to the input video containing pose data.
        image_path (str): Path to the reference image. If the filename indicates a video (e.g., ends with '4'),
                          a random frame is extracted.
        obj_path (str): Path prefix for object appearance images (e.g., "path/to/object_0.jpg").
        obj_track_path (str): Path to the video file for object trajectory reference.
        hand_path (str): Path to the video file for hand reference.
        resolution (int, optional): Target resolution for processing. Default is 576.
        sample_stride (int, optional): Frame sampling stride for video processing. Default is 2.
        num_frames (int, optional): Number of frames to process (-1 for all frames). Default is -1.
        
    Returns:
        tuple: Contains the following normalized tensors (values in the range [-1, 1]):
            - pose (torch.Tensor): Combined image and video pose tensor.
            - image (torch.Tensor): Processed reference image tensor.
            - object_appearance (torch.Tensor): Tensor of object appearance images.
            - object_trajectory (torch.Tensor): Tensor of object trajectory frames.
            - hand (torch.Tensor): Tensor of hand video frames.
    """
    # Load the reference image:
    if image_path[-1] == '4':
        # If the image_path ends with '4', treat it as a video file and extract a random frame.
        people_ref = decord.VideoReader(image_path, ctx=decord.cpu(0))
        idx = random.randrange(0, len(people_ref))
        people_ref = people_ref[idx].asnumpy()
        image_pixels = torch.from_numpy(people_ref).permute((2, 0, 1))  # (C, H, W)
    else:
        # Otherwise, load the image using a PIL loader.
        image_pixels = pil_loader(image_path)
        image_pixels = pil_to_tensor(image_pixels)

    # Get original image dimensions.
    h, w = image_pixels.shape[-2:]

    # Compute target dimensions based on the original aspect ratio.
    # The target dimensions are adjusted to be multiples of 64.
    if h > w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution

    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        # Set the height to the target height and adjust width accordingly.
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        # Set the width to the target width and adjust height accordingly.
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target

    # Resize and center-crop the image to the desired dimensions.
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()  # Convert to (H, W, C) numpy array

    # Retrieve pose values from the reference image and the input video.
    image_pose = get_image_pose(image_pixels)
    video_pose, p_scale, p_bias = get_video_pose(
        video_path,
        image_pixels,
        sample_stride=sample_stride,
        align_param=True,
        num_frames=num_frames
    )

    # Combine the image pose with the video pose.
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))  # Reshape to (1, C, H, W)

    if obj_path != "":
        # --- Object Appearance Reference ---
        obj_pixels = []
        idx = 0
        suffix = obj_path[-4:]  # e.g. ".jpg" or ".png"
        read_file = obj_path[:-5] + str(idx) + suffix

        while os.path.exists(read_file):
            # Load object appearance image, convert to RGB, and reformat to tensor (C, H, W)
            read_obj_pixels = torch.from_numpy(np.array(Image.open(read_file).convert("RGB"))).permute((2, 0, 1))

            # Pad the image to obtain a square shape.
            gap = (read_obj_pixels.shape[1] - read_obj_pixels.shape[2]) // 2
            padding = (max(0, gap), max(0, gap), max(0, -gap), max(0, -gap), 0, 0)
            read_obj_pixels = F.pad(read_obj_pixels, padding, mode='constant', value=0)

            # Resize the object image to 518x518.
            read_obj_pixels = resize(read_obj_pixels, [518, 518], antialias=None)
            obj_pixels.append(read_obj_pixels)

            idx += 1
            read_file = obj_path[:-5] + str(idx) + suffix

        # Stack the object images into a tensor with shape (num_images, C, H, W)
        obj_pixels = torch.stack(obj_pixels, dim=0)
        pic_num = obj_pixels.shape[0]
        # If fewer than 3 images are available, duplicate the first image.
        while pic_num < 3:
            obj_pixels = torch.cat([obj_pixels, obj_pixels[:1]], 0)
            pic_num += 1

        # --- Object Trajectory Reference ---
        obj_track_pixels = process_video_reference(obj_track_path, sample_stride, resolution, num_frames, p_scale, p_bias)

        # --- Hand Video Reference ---
        hand_pixels = process_video_reference(hand_path, sample_stride, resolution, num_frames, p_scale, p_bias)
    else:
        # If no object reference is provided, create black images as placeholders.
        black_pixels = torch.zeros_like(torch.from_numpy(pose_pixels.copy()), dtype=torch.float16)
        hand_pixels = black_pixels.numpy()
        obj_track_pixels = black_pixels.numpy()
        obj_pixels = torch.zeros((3, 3, 518, 518), dtype=torch.float16)

    # Normalize and return all outputs (scaling to the range [-1, 1]).
    return (torch.from_numpy(pose_pixels.copy()) / 127.5 - 1,
            torch.from_numpy(image_pixels) / 127.5 - 1,
            obj_pixels / 127.5 - 1,
            torch.from_numpy(obj_track_pixels) / 127.5 - 1,
            torch.from_numpy(hand_pixels / 127.5 - 1))

def run_pipeline(pipeline: AnchorCrafterPipeline,
                 image_pixels, pose_pixels, obj_pixels, obj_track_pixels, hand_pixels,
                 device, task_config):
    """
    Run the anchor crafter pipeline to synthesize video frames based on multiple inputs.
    Args:
        pipeline (AnchorCrafterPipeline): The pipeline instance used for video synthesis.
        image_pixels (Tensor): Normalized reference image tensor (values in [-1, 1]).
        pose_pixels (Tensor): Pose tensor with shape (N, ...), where N is the number of frames.
        obj_pixels (Tensor): Normalized object appearance tensor (values in [-1, 1]).
        obj_track_pixels (Tensor): Object trajectory tensor.
        hand_pixels (Tensor): Hand video tensor.
        device (torch.device): The computation device.
        task_config: Configuration object with parameters such as seed, num_frames,
                     frames_overlap, noise_aug_strength, num_inference_steps, guidance_scale, etc.
        args: Additional arguments (not used in this function).

    Returns:
        Tensor: Synthesized video frames as a uint8 tensor, with the first (deprecated) frame removed.
    """
    # Convert normalized image tensors to PIL images.
    # Scale from [-1, 1] to [0, 255] and convert each image to torch.uint8.
    image_pixels = [
        to_pil_image(img.to(torch.uint8))
        for img in ((image_pixels + 1.0) * 127.5)
    ]
    obj_pixels = [
        to_pil_image(img.to(torch.uint8))
        for img in ((obj_pixels + 1.0) * 127.5)
    ]

    # Set up a random generator with a fixed seed.
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)

    # Determine the minimum number of frames across the pose, object trajectory, and hand videos.
    frnum = min(pose_pixels.size(0), obj_track_pixels.size(0), hand_pixels.size(0))

    # Run the synthesis pipeline with the given inputs.
    frames = pipeline(
        image_pixels,
        pose_pixels[:frnum],
        obj_pixels,
        obj_track_pixels[:frnum],
        hand_pixels=hand_pixels[:frnum],
        num_frames=frnum,
        tile_size=task_config.num_frames,
        tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2],
        width=pose_pixels.shape[-1],
        fps=7,
        noise_aug_strength=task_config.noise_aug_strength,
        num_inference_steps=task_config.num_inference_steps,
        generator=generator,
        min_guidance_scale=task_config.guidance_scale,
        max_guidance_scale=task_config.guidance_scale,
        decode_chunk_size=8,
        output_type="pt",
        device=device,
    ).frames.cpu()

    # Convert synthesized frames from [0, 1] to [0, 255] and cast to uint8.
    video_frames = (frames * 255.0).to(torch.uint8)


    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames

@torch.no_grad()
def main(args):
    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)
    
    infer_config = OmegaConf.load(args.inference_config)
    
    pipeline = create_pipeline(infer_config, device)
    
    # Process each test case specified in the inference configuration.
    for task in infer_config.test_case:
        ##############################################
        # Pre-process data: load and process the reference video pose,
        # reference image, object appearance, object trajectory, and hand video.
        ##############################################
        pose_pixels, image_pixels, obj_pixels, obj_track_pixels, hand_pixels = preprocess(
            task.ref_video_path,
            task.ref_image_path,
            obj_path=task.obj_path,
            obj_track_path=task.obj_track_path,
            hand_path=task.hand_path,
            resolution=task.resolution,
            sample_stride=task.sample_stride,
            num_frames=args.num_frames,
        )
        
        ##############################################
        # Run the synthesis pipeline to generate video frames.
        ##############################################
        _video_frames = run_pipeline(
            pipeline,
            image_pixels,
            pose_pixels,
            obj_pixels,
            obj_track_pixels,
            hand_pixels=hand_pixels,
            device=device,
            task_config=task,
        )
        
        ##############################################
        # Save the results to the output folder in MP4 format.
        ##############################################
        base_video_name = os.path.basename(task.ref_video_path).split('.')[0]
        base_image_name = os.path.basename(task.ref_image_path).split('.')[0]
        output_filename = f"{args.output_dir}/{base_video_name}_{base_image_name}.mp4"
        
        save_to_mp4(
            _video_frames,
            output_filename,
            fps=task.fps,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml")  # ToDo
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
                        )
    parser.add_argument('--num_frames', type=int, default=-1)  # -1 means all frames

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
    logger.info(f"--- Finished ---")
