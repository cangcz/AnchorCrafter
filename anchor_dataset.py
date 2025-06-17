import math
import os
import random

import numpy as np

import torch
import torch.jit
from torch.utils.data import Dataset

from anchor_crafter.dwpose.preprocess import get_video_pose, get_image_pose

from constants import ASPECT_RATIO

from PIL import Image, ImageDraw, ImageFont
import decord
import numpy as np
import torch
from torchvision.transforms import Resize, CenterCrop
import random
import os
import torch.nn.functional as F
import concurrent.futures


def preprocess(video_path, pose_path, people_path=None, obj_path=None, box_path=None, hand_path=None, resolution=576,
               sample_stride=2, sample_frames=16, data_type=1):
    """Preprocess input data for training phase.
    
    Args:
        video_path: Path to the input video
        pose_path: Path to the pose video
        people_path: Path to reference people image/video
        obj_path: Path to object reference images
        box_path: Path to bounding box video
        hand_path: Path to hand video
        resolution: Target resolution for processing
        sample_stride: Frame sampling stride
        sample_frames: Number of frames to sample
        data_type: Processing mode ('base' or 'noobj')
    
    Returns:
        Tuple of processed tensors: (pose, people_ref, video, obj_ref, boxes, hands)
        All tensors are normalized to [-1, 1] range.
    """
    resize = Resize([int(resolution / ASPECT_RATIO), resolution])

    def center_crop(image, aspect_ratio=ASPECT_RATIO):
        """Center crop and resize image to maintain aspect ratio.
        
        Args:
            image: Input image (PIL Image or numpy array)
            aspect_ratio: Target aspect ratio
            
        Returns:
            Processed PIL Image
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        width, height = image.size
        if width == int(height * aspect_ratio) and height == resolution:
            return  image
        new_width = min(width, int(height * aspect_ratio))
        new_height = min(height, int(width / aspect_ratio))
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        image = image.crop((left, top, right, bottom))
        image = resize(image)
        return image

    def process_frames(video, frame_ids, add_text=False):
        """Process multiple frames.
        
        Args:
            video: VideoReader object
            frame_ids: List of frame indices to process
            add_text: Whether to add watermark text
            
        Returns:
            Tensor of processed frames in (T,C,H,W) format
        """
        frame = video.get_batch(frame_ids).asnumpy()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            cropped_frames = list(executor.map(center_crop, frame))
        cropped_frames = np.array(cropped_frames)

        if add_text:
            new_frames = []
            for frame in cropped_frames:
                new_frames.append(draw_text(frame))
            cropped_frames = np.stack(new_frames, axis=0)

        pixels = torch.from_numpy(cropped_frames).permute((0, 3, 1, 2))
        return pixels

    def draw_text(image):
        """Add watermark text to the image.
        
        Args:
            image: Input image as numpy array (H,W,C)
            
        Returns:
            Image with watermark as numpy array (H,W,C)
        """
        image = Image.fromarray(image)

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('Ubuntu-MI.ttf', 30)
        text = 'AnchorCrafter'

        # Get text bounding box
        left, top, right, bottom = draw.textbbox((0, 0), text, font) 
        text_width, text_height = right - left, bottom - top
        # Position at bottom-right corner
        x = image.size[0] - text_width
        y = image.size[1] - text_height

        # Draw text
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        # image.save('./draw.jpg')

        image = np.array(image)

        return image


    def process_single_frame(video, frame_id, add_text=False):
        """Process single frame from video.
        
        Args:
            video: VideoReader object
            frame_id: Frame index to process
            add_text: Whether to add watermark text
            
        Returns:
            Tensor of repeated frame in (T,C,H,W) format
        """
        frame = video[frame_id].asnumpy()
        cropped_frames = np.array(center_crop(frame))

        if add_text:
            cropped_frames = draw_text(cropped_frames)

        pixels = torch.from_numpy(cropped_frames).permute((2, 0, 1))
        return pixels.unsqueeze(0).repeat(sample_frames, 1, 1, 1)

    if data_type == 'base':
        # Load all required videos
        pose_video = decord.VideoReader(pose_path, ctx=decord.cpu(0))
        video_video = decord.VideoReader(video_path, ctx=decord.cpu(0))
        box_video = decord.VideoReader(box_path, ctx=decord.cpu(0))
        hand_video = decord.VideoReader(hand_path, ctx=decord.cpu(0))

        # Randomly select frame sequence
        start_idx = random.randint(0, min(len(pose_video), len(video_video), len(box_video),
                                          len(hand_video)) - sample_frames * sample_stride)
        frame_ids = list(range(start_idx, start_idx + sample_frames * sample_stride, sample_stride))
        # Process all frames
        pose_pixels = process_frames(pose_video, frame_ids)
        video_pixels = process_frames(video_video, frame_ids, add_text=False)
        box_video_pixels = process_frames(box_video, frame_ids)
        hand_pixels = process_frames(hand_video, frame_ids)

        # Process reference people image
        if people_path[-1] == '4':  # mp4
            people_ref = decord.VideoReader(people_path, ctx=decord.cpu(0))
            idx = random.randrange(0, len(people_ref))
            people_ref = people_ref[idx].asnumpy()

        else: 
            people_ref = Image.open(people_path).convert("RGB")

        people_ref = np.array(center_crop(people_ref))
        image_pixels = torch.from_numpy(people_ref).permute((2, 0, 1))
        image_pixels = image_pixels.unsqueeze(0)

        # Process object reference images
        obj_pixels = []
        idx = 0
        suffix = obj_path[-4:]
        read_file = obj_path[:-5] + str(idx) + suffix
        while (os.path.exists(read_file) and idx < 3):
            read_obj_pixels = torch.from_numpy(np.array(Image.open(read_file).convert("RGB"))).permute((2, 0, 1))
            gap = (read_obj_pixels.shape[1] - read_obj_pixels.shape[2]) // 2
            padding = (max(0, gap), max(0, gap), max(0, -gap), max(0, -gap), 0, 0)
            read_obj_pixels = F.pad(read_obj_pixels, padding, mode='constant', value=0)
            read_obj_pixels = Resize([518, 518])(read_obj_pixels)
            obj_pixels.append(read_obj_pixels)
            idx += 1
            read_file = obj_path[:-5] + str(idx) + suffix

        obj_pixels = torch.stack(obj_pixels, dim=0)
        while len(obj_pixels) < 3:
            obj_pixels = torch.cat([obj_pixels, obj_pixels[:1]], 0)

        # Special handling for first frame
        video_pixels = torch.cat([image_pixels, video_pixels], dim=0)
        image_pose = get_image_pose(people_ref)
        image_pose = torch.from_numpy(image_pose).unsqueeze(0)
        pose_pixels = torch.cat([image_pose, pose_pixels], dim=0)
        # Zero padding for first frame of boxes and hands
        obj_zero = torch.zeros(box_video_pixels[0].shape)
        box_video_pixels = torch.cat([obj_zero.unsqueeze(0), box_video_pixels], dim=0)
        hand_pixels = torch.cat([obj_zero.unsqueeze(0), hand_pixels], dim=0)
        return (pose_pixels / 127.5 - 1, 
                image_pixels / 127.5 - 1, 
                video_pixels / 127.5 - 1, 
                obj_pixels / 127.5 - 1,
                box_video_pixels / 127.5 - 1, 
                hand_pixels / 127.5 - 1)

    elif data_type == 'noobj':
        # Simplified processing without object references
        pose_video = decord.VideoReader(pose_path, ctx=decord.cpu(0))
        video_video = decord.VideoReader(video_path, ctx=decord.cpu(0))

        start_idx = random.randint(0, min(len(pose_video), len(video_video)) - sample_frames * sample_stride)
        frame_ids = list(range(start_idx, start_idx + sample_frames * sample_stride, sample_stride))

        pose_pixels = process_frames(pose_video, frame_ids)
        video_pixels = process_frames(video_video, frame_ids, add_text=False)

        # Use random frame from video as people reference
        frame_id = random.randint(0, len(video_video) - 1)
        people_pixels = process_single_frame(video_video, frame_id)[0:1]

        # Special handling for first frame
        video_pixels = torch.cat([people_pixels, video_pixels], dim=0)
        image_pose = get_image_pose(np.array(center_crop(video_video[frame_id].asnumpy())))
        image_pose = torch.from_numpy(image_pose).unsqueeze(0)
        pose_pixels = torch.cat([image_pose, pose_pixels], dim=0)

        # Empty object references and other inputs
        obj_pixels = torch.zeros((3,3,518,518), dtype=video_pixels.dtype)
        black_pixels = torch.zeros_like(video_pixels)

        return (pose_pixels / 127.5 - 1, 
                people_pixels / 127.5 - 1, 
                video_pixels / 127.5 - 1, 
                obj_pixels / 127.5 - 1,
                black_pixels / 127.5 - 1, 
                black_pixels / 127.5 - 1)


class AnchorDataset(Dataset):
    def __init__(self, base_folder: str, noobj_folder: str, resolution=576, sample_frames=16,
                 sample_stride=2):
        self.base_folder = base_folder
        self.noobj_folder = noobj_folder
        self.channels = 3
        self.resolution = resolution
        self.sample_frames = sample_frames
        self.sample_stride = sample_stride

        # Load samples from both folders
        self.base_samples = self._load_base_samples(base_folder, 'base')
        if noobj_folder is not "":
            self.noobj_samples_DreamDance = self._load_noobj_samples(noobj_folder, 'noobj')
        # self.sample = self.base_samples + self.noobj_samples_DreamDance
        self.sample = self.base_samples
        self.length = len(self.sample)
    def _load_base_samples(self, folder, type):
        samples = []
        for subfolder in os.listdir(folder):
            path = os.path.join(folder, subfolder)
            if os.path.isdir(path):
                files = os.listdir(os.path.join(path, "video_cut"))
                for file in files:
                    name = file.split(".")[0]
                    sample = {
                        'folder_path': path,
                        'people': name.split("_")[0],
                        'object': name.split("_")[1],
                        'clip': name.split("_")[2] if len(name.split("_")) > 2 else None,
                        'preprocess': type
                    }
                    samples.append(sample)
        return samples
    
    def _load_noobj_samples(self, folder, type):
        samples_DreamDance = []
        for subfolder in os.listdir(folder):
            path = os.path.join(folder, subfolder)
            if os.path.isdir(path):
                files = os.listdir(os.path.join(path, "video_cut"))
                for file in files:
                    sample = {
                        'folder_path': path,
                        'name': file,
                        'preprocess': type
                    }
                    samples_DreamDance.append(sample)

        return  samples_DreamDance

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        # ran = random.random()
        # if ran <= 0.80:
        #     sample=random.choice(self.base_samples)
        # else:
        #     sample = random.choice(self.noobj_samples_DreamDance)

        sample = self.sample[idx]
        print(sample)
        if sample['preprocess'] == 'base':
            video_name = sample['people'] + '_' + sample['object'] + (
                '_' + sample['clip'] if sample['clip'] is not None else '') + '.mp4'
            ref_video_path = os.path.join(sample['folder_path'], "video_cut", video_name)
            ref_people = os.path.join(sample['folder_path'], "people_cut", sample['people'])
            # Determine the reference people path
            if os.path.exists(ref_people + '_0.png'):
                ref_people_path = ref_people + '_0.png'
            elif os.path.exists(ref_people + '_0.jpg'):
                ref_people_path = ref_people + '_0.jpg'
            else:
                ref_people_path = ref_people + '.mp4'

            ref_pose_path = os.path.join(sample['folder_path'], "video_pose", video_name)
            ref_obj_path = os.path.join(sample['folder_path'], "masked_object_cut", sample['object'] + '_0')
            ref_box_path = os.path.join(sample['folder_path'], "depth_cut", video_name)

            if os.path.exists(ref_obj_path + '.jpg'):
                ref_obj_path = ref_obj_path + '.jpg'
            else:
                ref_obj_path = ref_obj_path + '.png'

            ref_hand_path = os.path.join(sample['folder_path'], "hand_cut", video_name)

        elif sample['preprocess'] == 'noobj':
            video_name = sample['name']
            ref_video_path = os.path.join(sample['folder_path'], "video_cut", video_name)
            ref_people_path = None
            ref_pose_path = os.path.join(sample['folder_path'], "pose_cut", video_name)
            ref_obj_path = None
            ref_box_path = None
            ref_hand_path = None

        pose_pixels, image_pixels, video_pixels, obj_pixels, box_video_pixels, hand_pixels = preprocess(
            ref_video_path,
            pose_path=ref_pose_path,
            people_path=ref_people_path,
            obj_path=ref_obj_path,
            box_path=ref_box_path,
            hand_path=ref_hand_path,
            resolution=self.resolution,
            sample_stride=self.sample_stride,
            sample_frames=self.sample_frames,
            data_type=sample['preprocess']
        )
        return {
            'pose_pixels': pose_pixels,
            'image_pixels': image_pixels,
            "video_pixels": video_pixels,
            'obj_pixels': obj_pixels,
            "box_video_pixels": box_video_pixels,
            'hand_pixels': hand_pixels
        }



