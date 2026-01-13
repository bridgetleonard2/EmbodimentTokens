from PIL import Image
import numpy as np
import urllib.request
import sys
from streamlit import json
from tqdm import tqdm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_vitpose", action="store_true", help="Use ViTPose for pose estimation")
args = parser.parse_args()

if args.use_vitpose:
    print("Using ViTPose for pose estimation")
    from transformers import (AutoProcessor,  # type: ignore
                            RTDetrForObjectDetection,
                            VitPoseForPoseEstimation)


# Helper functions
def load_image(image_id):
    image = Image.open(
        urllib.request.urlopen(
            f'http://images.cocodataset.org/train2017/{image_id}.jpg')
            ).convert("RGB")
    # make image 336x336
    image = image.resize((336, 336))
    return image


def preprocess_pose_data(pose_output):
    """Converts pose output to a flattened vector"""
    keypoints = pose_output["keypoints"]  # Shape: (17, 2)
    scores = pose_output["scores"]  # Shape: (17,)
    # select 'important' keypoints (shoulders,hips,scores)
    selected_indices = [5, 6, 11, 12]
    keypoints = keypoints[selected_indices]  # Shape: (4, 2)
    scores = scores[selected_indices]  # Shape: (4,)

    keypoints_flat_int = np.round(keypoints.flatten())  # Shape: (8,)

    scores_rounded = np.round(scores, 1)

    # For each keypoint append the score directly after the keypoint
    keypoints_scores = np.zeros((12,))

    keypoints_scores[0::3] = keypoints_flat_int[0::2]  # x-coordinates
    keypoints_scores[1::3] = keypoints_flat_int[1::2]  # y-coordinates
    keypoints_scores[2::3] = scores_rounded  # Corresponding scores
    keypoints_scores = np.round(keypoints_scores, 1)
    return keypoints_scores


def get_vit_keypoints(person_image_processor, person_model, image_processor, model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = person_image_processor(
        images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor(
            [(image.height, image.width)]), threshold=0.3
    )
    result = results[0]  # take first image results

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    inputs = image_processor(
        image, boxes=[person_boxes], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = image_processor.post_process_pose_estimation(
        outputs, boxes=[person_boxes])
    image_pose_result = pose_results[0]  # results for first image
    return image_pose_result


def get_coco_keypoints(coco_id, train_keypoints, target_size=336):
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    keypoints_list = []

    # Find annotation
    annotations = [ann for ann in train_keypoints["annotations"] if ann["image_id"] == coco_id]
    assert len(annotations) <= 1, "More than one person in image!"

    if len(annotations) == 1:
        ann = annotations[0]
        keypoints = ann['keypoints']

        # --- find original image dimensions ---
        images = train_keypoints["images"]

        image_info = next(img for img in images if img["id"] == coco_id)
        orig_w, orig_h = image_info["width"], image_info["height"]

        # --- scaling helper ---
        def scale_point(x, y):
            return (x * target_size / orig_w, y * target_size / orig_h)

        # --- shoulders & hips only ---
        for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]:
            x, y = keypoints[idx*3], keypoints[idx*3 + 1]
            v = keypoints[idx*3 + 2]  # keep visibility if you want
            if v > 0:
                x, y = scale_point(x, y)
                keypoints_list.extend([x, y])
            else:
                return None

        return [round(kp) for kp in keypoints_list]
    else:
        return None


# load coco items to process
data_items = np.load("../coco_orientations.npy")

if args.use_vitpose:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_image_processor = AutoProcessor.from_pretrained(
        "PekingU/rtdetr_r50vd_coco_o365")
    person_model = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd_coco_o365", device_map=device)

    image_processor = AutoProcessor.from_pretrained(
        "usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-base-simple", device_map=device)

    # initiate keypoints
    keypoints = np.zeros((len(data_items), 12))

    for i in tqdm(range(len(data_items))):
        image_id = data_items[i][0]
        image = load_image(image_id)
        try:
            keypoints_output = get_vit_keypoints(person_image_processor, person_model, image_processor, model, image)
            keypoints_output_cpu = {
                key: value.cpu().detach().numpy() for key,
                value in keypoints_output[0].items()}
            keypoints[i] = preprocess_pose_data(keypoints_output_cpu)
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            # add a placeholder np.array of nan values
            keypoints[i] = np.empty((1, 12)) * np.nan
            continue

        np.save(f"../vitpose/keypoints_data.npy", keypoints)
else:
    with open("r../coco/person_keypoints_train2017.json", "r") as f:
        train_keypoints = json.load(f)
    coco_ids = np.array([item[0] for item in data_items])
    coco_ids_int = coco_ids.astype(int)

    all_keypoints = []
    kept_cocos = []

    for coco_id in tqdm(coco_ids_int):
        kps = get_coco_keypoints(coco_id, train_keypoints)
        if kps is not None:
            all_keypoints.append(kps)
            kept_cocos.append(coco_id)

    all_keypoints = np.array(all_keypoints)
    kept_cocos = np.array(kept_cocos)

    # combine kept_cocos and all_keypoints
    combined = np.hstack((kept_cocos.reshape(-1, 1), all_keypoints[all_keypoints != None].reshape(-1, 8)))
    
    np.save(f'../coco/keypoints_data.npy', combined)
