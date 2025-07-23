import os
import cv2
import numpy as np

def calculate_pixel_similarity_ratio(reference_folder, generated_folder):
    """Calculate pixel-wise similarity ratio between reference and generated videos.
    
    Args:
        reference_folder: Path to directory containing reference videos
        generated_folder: Path to directory containing generated videos
    
    Returns:
        Mean similarity ratio across all valid video pairs
    """
    similarity_ratios = []  # Stores similarity ratios for each video

    # Get all video files in reference directory
    ref_videos = [f for f in os.listdir(reference_folder) 
                 if f.endswith(('.mp4', '.avi'))]

    for video_name in ref_videos:
        ref_path = os.path.join(reference_folder, video_name)
        gen_path = os.path.join(generated_folder, video_name)

        if not os.path.exists(gen_path):
            print(f"Generated video {gen_path} does not exist. Skipping.")
            continue

        # Initialize video capture objects
        ref_cap = cv2.VideoCapture(ref_path)
        gen_cap = cv2.VideoCapture(gen_path)
        
        total_same_pixels = 0   # Pixels non-zero and identical in both videos
        total_nonzero_pixels = 0  # Pixels non-zero in either video

        try:
            while True:
                # Read frames from both videos
                ref_ret, ref_frame = ref_cap.read()
                gen_ret, gen_frame = gen_cap.read()
                
                # Break if either video ends
                if not (ref_ret and gen_ret):
                    break

                # Convert to grayscale
                ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
                
                # Resize generated frame to match reference dimensions
                gen_resized = cv2.resize(gen_gray, (ref_gray.shape[1], ref_gray.shape[0]))
                
                # Calculate matching pixels (non-zero and identical)
                same_mask = (ref_gray != 0) & (gen_resized != 0) 
                same_pixels = np.sum(same_mask)
                
                # Calculate union of non-zero pixels
                nonzero_union = np.sum((ref_gray != 0) | (gen_resized != 0))
                
                total_same_pixels += same_pixels
                total_nonzero_pixels += nonzero_union

        finally:
            # Ensure resources are released
            ref_cap.release()
            gen_cap.release()

        # Calculate ratio for current video
        if total_nonzero_pixels > 0:
            video_ratio = total_same_pixels / total_nonzero_pixels
            similarity_ratios.append(video_ratio)

    # Return mean ratio if valid videos exist
    return np.mean(similarity_ratios) if similarity_ratios else 0.0

# Example usage
if __name__ == "__main__":
    REF_DIR = "./example/gt/mask/"
    GEN_DIR = "./example/results/mask/"
    
    similarity_ratio = calculate_pixel_similarity_ratio(REF_DIR, GEN_DIR)
    print(f"Pixel similarity ratio: {similarity_ratio:.4f}")
