from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


def get_video_pose(
        video_path: str, 
        ref_image: np.ndarray, 
        sample_stride: int=1,
        num_frames: int=-1,
        align_param: bool=False,
):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image 
        sample_stride (int, optional): Defaults to 1.

    Returns:
        np.ndarray: sequence of video pose
    """
    # select ref-keypoint from reference pose for pose rescale
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    # sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    if num_frames != -1:
        frames = frames[:num_frames, ...]
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
    if align_param:
        return np.stack(output_pose), a, b
    return np.stack(output_pose)


def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)


def align_pic(image, p_scale, p_bias, input_format="channel_first"):
    """
    Aligns the input image using scaling and bias parameters.
    
    If the input is in channel-first format (C, H, W), it will be converted
    to channel-last (H, W, C) internally, processed, and then converted back.
    
    Args:
        image (np.ndarray): Input image.
        p_scale (tuple): Scaling factors (scale_y, scale_x).
        p_bias (tuple): Bias values (bias_y, bias_x).
        input_format (str, optional): "channel_first" or "channel_last".
        
    Returns:
        np.ndarray: Aligned image in the original format.
    """
    # If the input is in channel_first format, transpose it to channel_last for processing.
    if input_format == "channel_first":
        image = np.transpose(image, (1, 2, 0))
        revert_format = True
    else:
        revert_format = False

    # Get image dimensions.
    height, width, _ = image.shape
    scale_y, scale_x = p_scale
    bias_y, bias_x = p_bias

    # Compute the absolute offsets based on the image dimensions.
    offset_x = bias_x * height
    offset_y = bias_y * width

    # Generate the coordinate grid for the target image.
    i_grid, j_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Compute the original image coordinates using the inverse affine transformation.
    x_original = ((i_grid - offset_x) / scale_x).astype(int)
    y_original = ((j_grid - offset_y) / scale_y).astype(int)

    # Create a mask to avoid index out-of-bound errors.
    mask = (0 <= x_original) & (x_original < height) & (0 <= y_original) & (y_original < width)

    # Create a new image and copy valid pixels from the original image.
    aligned_image = np.zeros_like(image)
    aligned_image[mask] = image[x_original[mask], y_original[mask]]

    # If the original input was in channel_first format, transpose the result back.
    if revert_format:
        aligned_image = np.transpose(aligned_image, (2, 0, 1))
    return aligned_image


def align_track_video(video, p_scale, p_bias):
    """
    Aligns all frames of the input video.
    
    Args:
        video (np.ndarray): Video tensor with shape (F, C, H, W).
        p_scale (tuple): Scale parameters.
        p_bias (tuple): Bias parameters.
        
    Returns:
        np.ndarray: Aligned video with shape (F, C, H, W).
    """
    aligned_frames = []
    for frame in video:
        new_frame = align_pic(frame, p_scale, p_bias, input_format="channel_first")
        aligned_frames.append(new_frame)

    aligned_video = np.stack(aligned_frames, axis=0)  # (F, C, H, W)
    return aligned_video