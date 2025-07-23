import cv2
import os
import numpy as np

path_groups = [
    {
        "mask_folder": "./example/results/mask",
        "video_folder": "./example/results/",
        "output_folder": "./example/results/object/"
    },
    {
        "mask_folder": "./example/gt/mask",
        "video_folder": "./example/gt/",
        "output_folder": "./example/gt/object/"
    }
]

for group in path_groups:
    mask_folder = group["mask_folder"]
    video_folder = group["video_folder"]
    output_folder = group["output_folder"]
    
    # Traverse the mask file
    for mask_filename in os.listdir(mask_folder):
        if mask_filename.endswith(".mp4"):
            # Construct the corresponding path of the original video file
            video_path = os.path.join(video_folder, mask_filename)
            mask_path = os.path.join(mask_folder, mask_filename)
            output_path = os.path.join(output_folder, mask_filename.replace('.mp4', ''))
            
            os.makedirs(output_path, exist_ok=True)

            cap_video = cv2.VideoCapture(video_path)
            cap_mask = cv2.VideoCapture(mask_path)
            
            # Verify frame consistency
            if int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT)) != int(cap_mask.get(cv2.CAP_PROP_FRAME_COUNT)):
                print(f"Frames mismatch: {video_path}")
                print(f"Video frames: {int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))}")
                print(f"Mask frames: {int(cap_mask.get(cv2.CAP_PROP_FRAME_COUNT))}")
                continue

            frame_idx = 1
            while cap_video.isOpened() and cap_mask.isOpened():
                ret_video, frame_video = cap_video.read()
                ret_mask, frame_mask = cap_mask.read()

                if not ret_video or not ret_mask:
                    break

                # Adjust the mask size and binarize it
                frame_mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
                _, mask_binary = cv2.threshold(frame_mask_gray, 127, 255, cv2.THRESH_BINARY)
                mask_binary = cv2.resize(mask_binary, (frame_video.shape[1], frame_video.shape[0]))

                # Generate object region
                object_frame = cv2.bitwise_and(frame_video, frame_video, mask=mask_binary)
                
                # Find contours and save the results
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                output_filename = f"{frame_idx:05d}.jpg"
                save_path = os.path.join(output_path, output_filename)

                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    cv2.imwrite(save_path, object_frame[y:y+h, x:x+w])
                else:
                    cv2.imwrite(save_path, np.zeros((100, 100, 3), dtype=np.uint8))
                
                frame_idx += 1

            # Release resources
            cap_video.release()
            cap_mask.release()

print("finish!")