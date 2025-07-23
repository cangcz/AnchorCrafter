import os
import numpy as np

folder1 = "./example/results/hand/"
folder2 = "./example/gt/hand/"

# Get all the .npy file names in folder1
files1 = sorted([f for f in os.listdir(folder1) if f.endswith(".npy")])

# Store LMD calculation results
lmd_results = []

# Calculate the scaling ratio
scale_x = 1024 / 1920
scale_y = 576 / 1080

# Traverse the files in folder1 and search for the same file name in folder2
for file1 in files1:
    path1 = os.path.join(folder1, file1)
    path2 = os.path.join(folder2, file1)  # Match the file name directly

    if os.path.exists(path2):  # Ensure that the file exists
        keypoints1 = np.load(path1)
        keypoints2 = np.load(path2)
        if len(keypoints1)==0 or len(keypoints2)==0 or len(keypoints1)>2 or len(keypoints2)>2:
            if len(keypoints1)!=0:
                keypoints2 = np.zeros_like(keypoints1)
            elif len(keypoints2)!=0:
                keypoints1 = np.zeros_like(keypoints2)
            else:
                print("keypoints1 or keypoints2 is empty")
                continue

        keypoints2 = keypoints2.astype(np.float64)  # to float64
        keypoints2[:, :, 0] *= scale_x
        keypoints2[:, :, 1] *= scale_y

        # calc LMD
        lmd = np.sqrt(np.sum((keypoints1 - keypoints2) ** 2, axis=-1))

        lmd = lmd[lmd < 100]
        print(lmd)
        if lmd.size == 0:
            print("lmd is empty")
            continue
        lmd_results.append((file1, lmd.mean()))
        
# Output the matched LMD results
for f1, lmd in lmd_results:
    print(f"{f1} -> LMD: {lmd}")
ans = [lmd[1] for lmd in lmd_results]  # Extract the second element (LMD value)
ans_array = np.array(ans)
print(ans_array.mean())


