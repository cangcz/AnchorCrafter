import os
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection
import torch
from PIL import Image

import os
import re
from PIL import Image

def eval(generated_images_folder, reference_images_folder, vid, model, processor, device):
    generated_images = sorted(os.listdir(generated_images_folder))
    count = 0
    total_score = 0

    for gen_img_name in generated_images:
        # Extract the prefix (video ID)
        match = re.match(r"(\d{5})\.jpg", gen_img_name)
        if match:
            frame_number = match.group(1)      # frame ID
            ref_img_name = f"{frame_number}.jpg"
        else:
            print(f"file name format mismatch: {gen_img_name}, skipping")
            continue

        gen_img_path = os.path.join(generated_images_folder, gen_img_name)
        ref_img_path = os.path.join(reference_images_folder, ref_img_name)

        if not os.path.exists(ref_img_path):
            print(f"ref img {ref_img_name} not exist, skip {gen_img_name}")
            continue

        image1 = Image.open(gen_img_path)
        image2 = Image.open(ref_img_path)
        inputs = processor(images=[image1, image2], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds

        similarity = torch.nn.functional.cosine_similarity(image_features[0], image_features[1], dim=0).item()

        count += 1
        # CLIP Score
        total_score += similarity
        print(f"{gen_img_name} vs {ref_img_name} CLIP Score: {similarity}")
    average_score = total_score / count if count > 0 else 0
    print(vid + f" avg CLIP Score: {average_score}")

    return average_score


device = "cuda" if torch.cuda.is_available() else "cpu"

# load clip
model = CLIPVisionModelWithProjection.from_pretrained("./models/clip_ViT-B_32").to(device)
processor = CLIPProcessor.from_pretrained("./models/clip_ViT-B_32")
file_dir="./example/results/object/"

scores = dict()
for vid in os.listdir(file_dir):
    generated_images_folder = file_dir + vid
    reference_images_folder = "./example/gt/object/" + vid

    avg_score = eval(generated_images_folder, reference_images_folder, vid,model=model, processor=processor, device=device)
    scores[vid] = avg_score

print('avg CLIP Score: ', scores)
import numpy as np
scores_list = list(scores.values())
mean_score = np.mean(scores_list)    # Compute the mean
print('avg CLIP Score: ', mean_score)
