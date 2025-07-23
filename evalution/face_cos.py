import sys
sys.path.append('./adaface')
import net
import torch
import os
from face_alignment import align
import numpy as np
import argparse
import re

adaface_models = {
    'ir_50': "models/adaface_ir50_ms1mv2.ckpt",
}


def load_pretrained_model(architecture='ir_50'):
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    bgr_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([bgr_img.transpose(2, 0, 1)]).float()
    return tensor



def get_aligned_rgb_img_or_black(path):
    aligned_rgb_img = align.get_aligned_face(path)

    if aligned_rgb_img is None or aligned_rgb_img.size == 0:  # Ensure that the image is not empty or has a size of 0
        black_image = np.zeros((112, 112, 3), dtype=np.uint8)  # Adjust the size of the black image as needed
        return black_image

    return aligned_rgb_img



def calculate_average_similarity(model, image_paths, folder_paths):
    similarities = []
    for frame in os.listdir(folder_paths):
        match = re.match(r"(\d+)_\d+\.png", frame)
        extracted_data = match.group(1)
        image_path = os.path.join(image_paths,extracted_data)+'.jpg'
        print(image_path)
        path=os.path.join(folder_paths, frame)
        aligned_rgb_img = align.get_aligned_face(image_path)
        input_tensor = to_input(aligned_rgb_img)
        target_feature, _ = model(input_tensor)
        print(path)

        aligned_rgb_img = get_aligned_rgb_img_or_black(path)
        if aligned_rgb_img is None or aligned_rgb_img.size == 0:
            print("no_face!!!!!")
            continue

        input_tensor = to_input(aligned_rgb_img)
        feature, _ = model(input_tensor)
        similarity = torch.matmul(target_feature, feature.T).item()
        similarities.append(similarity)
        print(f"Similarity: {similarity}")
    print('over')
    average_similarity = np.mean(similarities)
    return average_similarity


if __name__ == '__main__':
    model = load_pretrained_model('ir_50')
    image_paths = './example/gt/human' # ref human image paths
    folder_paths = './example/results/all_frame' # frames
    average_similarity = calculate_average_similarity(model, image_paths, folder_paths)
    print(f"Average similarity: {average_similarity}")
