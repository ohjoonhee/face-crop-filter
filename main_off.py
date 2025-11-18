from datasets import load_dataset

import torch
from datasets import load_dataset, Dataset
from ultralytics import YOLO
from PIL import Image
import json
import numpy as np
import os
import os.path as osp
from tqdm.auto import tqdm


def main():
    repo_id = "ohjoonhee/custom1"
    model_path = "yolov12n-face.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    margin_scale = 0.3  # 20% margin

    ds = load_dataset(repo_id, split="train")

    BATCH_SIZE = 512
    JSON_OUTPUT = "crop_regions_margin30.jsonl"

    if osp.dirname(JSON_OUTPUT) != "":
        os.makedirs(osp.dirname(JSON_OUTPUT), exist_ok=True)

    model = YOLO(model_path)
    model.to(device)

    for bi in tqdm(range(0, len(ds), BATCH_SIZE)):
        images = ds[bi : bi + BATCH_SIZE]["image"]
        results = model(images, verbose=False, device=device)

        for i, result in enumerate(results):
            original_img = images[i]
            boxes = result.boxes

            # Check if any face was detected
            if len(boxes) == 0:
                e = {
                    "idx": bi + i,
                    "crop_region": None,
                }
                with open(JSON_OUTPUT, "a") as f:
                    f.write(json.dumps(e) + "\n")
                # cropped_images.append(None)  # Mark for removal later
                continue

            # Logic to find the 'Largest' face
            # boxes.xyxy returns [x1, y1, x2, y2] tensor
            # Calculate areas: (x2 - x1) * (y2 - y1)
            xyxy = boxes.xyxy
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]
            areas = widths * heights

            # Find index of the face with the maximum area
            max_area_idx = torch.argmax(areas).item()

            # Get coordinates of the largest face
            best_box = xyxy[max_area_idx].cpu().numpy()
            x1, y1, x2, y2 = best_box

            # --- Adaptive Margin Logic ---
            # Get image dimensions for boundary safety checks
            img_w, img_h = original_img.size

            # Calculate margin based on bbox size
            box_w = x2 - x1
            box_h = y2 - y1

            x_pad = box_w * margin_scale
            y_pad = box_h * margin_scale

            # Apply margin and Clamp to image boundaries (Safety Mechanism)
            x1 = max(0, int(x1 - x_pad))
            y1 = max(0, int(y1 - y_pad))
            x2 = min(img_w, int(x2 + x_pad))
            y2 = min(img_h, int(y2 + y_pad))

            # Save the crop region in json
            e = {
                "idx": bi + i,
                "crop_region": [x1, y1, x2, y2],
            }
            with open(JSON_OUTPUT, "a") as f:
                f.write(json.dumps(e) + "\n")

    with open(JSON_OUTPUT, "r") as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    lines = sorted(lines, key=lambda x: x["idx"])

    def crop_image(example, idx):
        crop_region = lines[idx]["crop_region"]
        if crop_region is None:
            return example["image"]
        x1, y1, x2, y2 = crop_region
        img = example["image"]
        cropped_img = img.crop((x1, y1, x2, y2))
        example["image"] = cropped_img
        return example

    new_ds = ds.map(
        crop_image,
        with_indices=True,
        num_proc=16,
    )

    new_ds.push_to_hub("ohjoonhee/custom1-cropped")


if __name__ == "__main__":
    main()
