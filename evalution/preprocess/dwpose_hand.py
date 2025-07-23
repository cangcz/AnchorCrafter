import torch
import torch.multiprocessing as mp
import os
import cv2
import numpy as np
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('./models/body_pose_model.pth')
hand_estimation = Hand('./models/hand_pose_model.pth')

def process_image(images):
    for image in images:
        test_image = os.path.join(frame_dir, image)
        oriImg = cv2.imread(test_image)
        if oriImg is None:
            continue

        candidate, subset = body_estimation(oriImg)
        hands_list = util.handDetect(candidate, subset, oriImg)

        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            cropped_hand = oriImg[y:y+w, x:x+w, :]
            peaks = hand_estimation(cropped_hand)
            
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)

            all_hand_peaks.append(peaks)

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, image.split('.')[0] + '.npy'), all_hand_peaks)

if __name__ == "__main__":
    path_groups = [
        {
            "frame_dir": "./example/gt/all_frame/",
            "out_dir": "./example/gt/hand/"
        },
        {
            "frame_dir": "./example/results/all_frame/",
            "out_dir": "./example/results/hand/"
        }
    ]

    for group in path_groups:
        global frame_dir, out_dir
        frame_dir = group["frame_dir"]
        out_dir = group["out_dir"]
        
        images = os.listdir(frame_dir)
        process_image(images)
        print(f"finish: {frame_dir} -> {out_dir}")

print("all finish")