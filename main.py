import torch
from datasets import load_dataset, Dataset
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os


def process_dataset_faces(
    dataset_name: str, split: str = "train", model_path: str = "yolov12n-face.pt", batch_size: int = 32, num_proc: int = 1, margin_scale: float = 0.2  # Added: default 20% margin
):
    """
    Loads a HF dataset, detects faces using YOLO, crops the largest face,
    and returns a new dataset.
    """

    # 1. Setup Device and Model
    # Use GPU if available for maximum efficiency
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO model on {device}...")

    # Load the YapaLab/YOLO-face model.
    # Ensure 'yolov8n-face.pt' is in your working directory or provided path.
    try:
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'yolov8n-face.pt' is downloaded from https://github.com/YapaLab/yolo-face/releases")
        return

    # 2. Load the Dataset
    print(f"Loading dataset '{dataset_name}'...")
    # usage: load_dataset("username/dataset_name", split="train")
    # Using a placeholder 'beans' dataset for demonstration if not specified
    if dataset_name == "demo_placeholder":
        dataset = load_dataset("beans", split=split)  # Just for testing structure
        print("Loaded demo dataset 'beans'. Replace with your actual face dataset.")
    else:
        dataset = load_dataset(dataset_name, split=split)

    # 3. Define the Batch Processing Function
    def batch_detect_and_crop(batch):
        """
        This function runs once per batch.
        Input: A dict {'image': [PIL.Image, ...], 'id': [...], ...}
        Output: A dict with the processed 'image' column.
        """
        images = batch["image"]

        # Ensure images are RGB (Hugging Face sometimes has Grayscale/RGBA)
        images = [img.convert("RGB") for img in images]

        # Run YOLO inference in batch
        # stream=False ensures we get all results at once
        # verbose=False keeps the console clean
        results = model(images, verbose=False, device=device)

        cropped_images = []

        for i, result in enumerate(results):
            original_img = images[i]
            boxes = result.boxes

            # Check if any face was detected
            if len(boxes) == 0:
                cropped_images.append(None)  # Mark for removal later
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

            # Crop the face
            face_crop = original_img.crop((x1, y1, x2, y2))
            cropped_images.append(face_crop)

        batch["image"] = cropped_images
        return batch

    # 4. Apply Map with Batching
    print("Starting batch processing...")
    # We use batched=True to send lists of images to YOLO.
    # num_proc=1 is recommended with GPU usage to avoid CUDA forking errors.
    # If using CPU only, you can increase num_proc (e.g., 4 or 8).
    processed_dataset = dataset.map(batch_detect_and_crop, batched=True, batch_size=batch_size, num_proc=None if device == "cuda" else num_proc, desc="Detecting and Cropping Faces")

    # 5. Filter out images where no face was detected (None values)
    print("Filtering out images with no faces...")
    final_dataset = processed_dataset.filter(lambda example: example["image"] is not None)

    print(f"Processing complete.")
    print(f"Original count: {len(dataset)}")
    print(f"Final count (faces detected): {len(final_dataset)}")

    return final_dataset


if __name__ == "__main__":
    # --- Configuration ---
    # Replace with your target dataset from Hugging Face Hub
    # Example: "celeb_a", "wider_face", or your private dataset
    TARGET_DATASET = "ohjoonhee/custom1"

    # Download the model weight first if you haven't:
    # !wget https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt

    new_dataset = process_dataset_faces(
        dataset_name=TARGET_DATASET,
        split="train",
        batch_size=256,
        num_proc=4,
        margin_scale=0.2,
    )  # Adjust based on your GPU VRAM  # Use >1 only if using CPU  # Example: 30% margin

    # Optional: Save the new dataset to disk
    # new_dataset.save_to_disk("cropped_faces_dataset")

    # Optional: Push to Hugging Face Hub
    new_dataset.push_to_hub(TARGET_DATASET + "-CF")
