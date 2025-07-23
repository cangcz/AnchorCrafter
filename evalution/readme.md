#  Evaluation
## Directory Structure


## Processing Pipeline
Place the generated videos and real videos under `example/gt` and `example/results`, respectively, then use [SAM2](https://sam2.metademolab.com/) to extract the object masks and save them in the corresponding `mask` folders. Place the Reference human images in `example/human`.
```plaintext
./evaluation
└──example/
    ├── gt/                  # Source content
    │   ├── 1.mp4      
    │   ├── 2.mp4  
    │   ├── ...     
    │   ├──mask/             # SAM-generated masks
    │   │   ├── 1.mp4
    │   │   └── 2.mp4
    │   └── human/           # Reference human image
    │        ├── 1.jpg
    │        └── 2.jpg      
    └── results/             # Generated content
        ├── 1.mp4        
        ├── 2.mp4
        ├── ...
        └── mask/            # SAM-generated masks
            ├── 1.mp4
            └── 2.mp4
```
### Step 1: Data Preparation
Install ffmpeg
```bash
cd evaluation
# Convert videos to frames
sh preprocess/video2frame.sh

# Extract objects using masks
python preprocess/extract_obj.py 
```
Refer to the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) repository for setup instructions.

Download [body_pose_model.pth, hand_pose_model.pth](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG) checkpoint to `models`.
```
# Extract hand poses 
python preprocess/dwpose_hand.py 
```
### Step 2: Evaluation Metrics

| Metric | Script | 
|--------|--------|
| Object IoU | `obj_iou.py` |
| CLIP Similarity | `obj_clip.py` | 
| Hand Landmark | `hand_lmd.py` | 
| Face Similarity | `face_cos.py` | 
1. Object IoU
```
python obj_iou.py
```
2. CLIP Similarity

Download [clip model](https://huggingface.co/sentence-transformers/clip-ViT-B-32/tree/main/0_CLIPModel) to `models/clip_ViT-B_32`.
```
python obj_clip.py
```
3. Hand Landmark
```
python hand_lmd.py
```
4. Face Similarity

Refer to the [AdaFace](https://github.com/mk-minchul/AdaFace) repository for setup instructions.
The file face_cos.py provides a reference implementation.
```
python face_cos.py
```


## Notes
```plaintext
./evaluation
├── example/
│   ├── gt/                  # Source content
│   │   ├── *.mp4            # Original driving videos
│   │   ├── all_frame/       # Extracted frames (PNG)
│   │   ├── mask/            # SAM-generated masks
│   │   ├── object/          # Cropped objects
│   │   ├── human/           # Reference human image
│   │   └── hand/            # OpenPose hands
│   └── results/             # Generated content
│       └── [same structure as gt/]
├── models/
│   ├── body_pose_model.pth  # OpenPose body model
│   ├── hand_pose_model.pth  # OpenPose hand model
│   └── clip_ViT-B_32        # clip
├── adaface/            # Adaface repository
└── *.py                # All metric calculation 
```
**File Naming Convention**:
   - Videos: `{video_id}.mp4`
   - Frames: `{video_id}_{n:04d}.png`
   - Masks: `{video_id}.mp4`
   - Human: `{video_id}.jpg`
