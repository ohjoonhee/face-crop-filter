import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, Features, Image as HFImage
from ultralytics import YOLO
from PIL import Image
import numpy as np


def process_dataset_optimized(
    dataset_name: str,
    split: str = "train",
    model_path: str = "yolov12n-face.pt",
    batch_size: int = 32,
    num_workers: int = 4,  # Number of CPU threads for data loading
    margin_scale: float = 0.2,
):
    # 1. Setup Device and Model (Load ONCE in the main process)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO model on {device}...")

    try:
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load the Source Dataset
    print(f"Loading source dataset '{dataset_name}'...")
    if dataset_name == "demo_placeholder":
        dataset = load_dataset("beans", split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    # Capture features to preserve schema (excluding 'image' if we wanted to change it, but we keep it as Image)
    original_features = dataset.features

    # 3. Create a DataLoader for Efficient Pipeline
    # This allows 'num_workers' CPUs to decode PIL images while the GPU is busy.
    def collate_fn(batch):
        # batch is a list of dicts: [{'image': PIL, 'id': 123, 'label': 'cat'}, ...]
        # We need to extract images for YOLO, but keep the rest of the data to yield later.
        images = [item["image"].convert("RGB") for item in batch]
        return images, batch

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=num_workers,
        num_workers=0,
        collate_fn=collate_fn,
        # prefetch_factor=2,  # Buffer batches to ensure GPU never waits
        # pin_memory=True if device == "cuda" else False,
    )

    # 4. Define the Generator
    def data_generator():
        print(f"Starting inference with {num_workers} CPU workers and 1 GPU process...")

        # Iterate through the pre-fetched batches
        for batch_images, batch_records in data_loader:
            # Run Inference (Batch Processing)
            # verbose=False speeds up the loop slightly
            results = model(batch_images, verbose=False, device=device)

            # Process Results
            for i, result in enumerate(results):
                boxes = result.boxes

                if len(boxes) == 0:
                    continue  # Skip images with no faces

                # Logic to find the largest face
                xyxy = boxes.xyxy
                widths = xyxy[:, 2] - xyxy[:, 0]
                heights = xyxy[:, 3] - xyxy[:, 1]
                areas = widths * heights

                max_area_idx = torch.argmax(areas).item()
                best_box = xyxy[max_area_idx].cpu().numpy()
                x1, y1, x2, y2 = best_box

                # --- Adaptive Margin Logic ---
                original_img = batch_images[i]
                img_w, img_h = original_img.size

                box_w = x2 - x1
                box_h = y2 - y1

                x_pad = box_w * margin_scale
                y_pad = box_h * margin_scale

                # Clamp coordinates
                x1 = max(0, int(x1 - x_pad))
                y1 = max(0, int(y1 - y_pad))
                x2 = min(img_w, int(x2 + x_pad))
                y2 = min(img_h, int(y2 + y_pad))

                # Crop
                face_crop = original_img.crop((x1, y1, x2, y2))

                # Get the original record and update the image
                record = batch_records[i]
                record["image"] = face_crop

                # Yield the complete record with other columns preserved
                yield record

    # 5. Build Dataset from Generator
    # We pass the original features to ensure columns like 'label', 'id' etc are preserved correctly
    print("Building new dataset from generator stream...")

    new_dataset = Dataset.from_generator(data_generator, features=original_features)

    return new_dataset


if __name__ == "__main__":
    # Configuration
    TARGET_DATASET = "ohjoonhee/custom1"

    final_dataset = process_dataset_optimized(
        dataset_name=TARGET_DATASET,
        split="train[:1024]",
        batch_size=64,
        num_workers=8,
        margin_scale=0.3,
    )  # Maximize this based on your VRAM  # Typical rule of thumb: 4-8 workers

    print(f"Done! New dataset has {len(final_dataset)} images.")
    # final_dataset.save_to_disk("./processed_faces")
