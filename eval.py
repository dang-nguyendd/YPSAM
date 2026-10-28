import os
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from segment_anything import SamPredictor, sam_model_registry
from constants import MODEL
from functions import InferenceSaver

# ==========================================
# CONFIGURATION
# ==========================================
RUN_VERSION = 2
DATA = "Kvasir_CVC-ClinicDB"
EPOCH = 13
MODEL_PATH = f"./model_pth/YOLOSAM_v1_run{RUN_VERSION}_{DATA}/YOLOSAM_v1_run{RUN_VERSION}_{DATA}{EPOCH}-last.pth"
VISUALS_DIR = Path(f"./results_data_{DATA}/visuals")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def find_mask_path(gt_root: str, image_name: str) -> str:
    """Finds the corresponding mask file even if its extension differs from the image."""
    stem = Path(image_name).stem
    exact_path = os.path.join(gt_root, image_name)
    if os.path.exists(exact_path):
        return exact_path

    for ext in VALID_EXTS:
        for candidate in [stem + ext, stem + ext.upper()]:
            cand_path = os.path.join(gt_root, candidate)
            if os.path.exists(cand_path):
                return cand_path
    raise FileNotFoundError(f"Mask not found for {image_name} in {gt_root}")


def load_binary_mask(mask_path: str) -> np.ndarray:
    """Loads mask and safely binarizes it regardless of [0,1] or [0,255] encoding."""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    max_val = mask.max()
    if max_val == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    return (mask > (max_val * 0.5)).astype(np.uint8)


def draw_yolo_boxes(ax, prompt_box):
    """Draws [x1, y1, x2, y2] bounding boxes onto a Matplotlib axis."""
    if prompt_box is None:
        return
    boxes = np.array(prompt_box)
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor="yellow", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)


def save_eval_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    prompt_box,
    image_name: str,
    dice: float,
    iou: float,
    output_dir: Path,
):
    """Exports a 4-panel Matplotlib figure for a single evaluation sample."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    h, w = gt_mask.shape

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"{image_name}  |  Dice: {dice:.4f}  |  IoU: {iou:.4f}",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: Raw Image + YOLO Box
    axes[0].imshow(image)
    draw_yolo_boxes(axes[0], prompt_box)
    box_status = "Detected" if prompt_box is not None else "MISSED"
    axes[0].set_title(f"Image + YOLO Box ({box_status})")
    axes[0].axis("off")

    # Panel 2: Ground Truth Mask
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Panel 3: Output Segmentation Mask
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("SAM Output Segmentation")
    axes[2].axis("off")

    # Panel 4: Overlay (Image + YOLO Box + TP/FP/FN Segmentation Map)
    axes[3].imshow(image)
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    # True Positive (Green), False Positive (Red), False Negative (Blue)
    overlay[(pred_mask == 1) & (gt_mask == 1)] = [0.0, 1.0, 0.0, 0.45]
    overlay[(pred_mask == 1) & (gt_mask == 0)] = [1.0, 0.0, 0.0, 0.45]
    overlay[(pred_mask == 0) & (gt_mask == 1)] = [0.0, 0.4, 1.0, 0.45]
    axes[3].imshow(overlay)
    draw_yolo_boxes(axes[3], prompt_box)
    axes[3].set_title("Overlay (Grn:TP | Red:FP | Blu:FN)")
    axes[3].axis("off")

    out_path = output_dir / f"{stem}_eval.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def test(model, bbox_coords_test, test_files, output_dir: Path, include_missed_detections=True):
    image_root = f"./data/{DATA}/images/test"
    gt_root    = f"./data/{DATA}/masks/test"

    model.eval()
    predictor_tuned = SamPredictor(model)

    images_path_list = sorted(test_files)
    num_images = len(images_path_list)

    if num_images == 0:
        raise ValueError("No test images to evaluate!")

    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    smooth = 1e-6

    with torch.no_grad():
        for idx, image_name in enumerate(images_path_list, 1):
            img_path = os.path.join(image_root, image_name)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(img_path)
            image = cv2.cvtColor(image, cv2.cvtColor_BGR2RGB if hasattr(cv2, 'cvtColor_BGR2RGB') else cv2.COLOR_BGR2RGB)

            mask_path = find_mask_path(gt_root, image_name)
            gt_mask = load_binary_mask(mask_path)
            h, w = gt_mask.shape

            prompt_box = bbox_coords_test.get(image_name, None)

            if prompt_box is None:
                if not include_missed_detections:
                    continue
                pred_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                predictor_tuned.set_image(image)
                prompt_box_np = np.array(prompt_box)

                if prompt_box_np.ndim == 1:
                    pred, _, _ = predictor_tuned.predict(
                        point_coords=None,
                        box=prompt_box_np,
                        multimask_output=False,
                    )
                    pred_mask = (pred[0] > 0).astype(np.uint8)
                else:
                    combined_mask = np.zeros((h, w), dtype=np.uint8)
                    for single_box in prompt_box_np:
                        pred, _, _ = predictor_tuned.predict(
                            point_coords=None,
                            box=single_box,
                            multimask_output=False,
                        )
                        combined_mask = np.logical_or(combined_mask, pred[0] > 0)
                    pred_mask = combined_mask.astype(np.uint8)

            # Flatten to 1D for confusion matrix
            pred_flat = pred_mask.reshape(-1)
            gt_flat = gt_mask.reshape(-1)

            TP = np.sum((pred_flat == 1) & (gt_flat == 1))
            FP = np.sum((pred_flat == 1) & (gt_flat == 0))
            FN = np.sum((pred_flat == 0) & (gt_flat == 1))

            dice = (2.0 * TP + smooth) / (2.0 * TP + FP + FN + smooth)
            iou = (TP + smooth) / (TP + FP + FN + smooth)
            precision = (TP + smooth) / (TP + FP + smooth)
            recall = (TP + smooth) / (TP + FN + smooth)

            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall

            # Export visual comparison figure
            save_eval_visualization(
                image=image,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                prompt_box=prompt_box,
                image_name=image_name,
                dice=dice,
                iou=iou,
                output_dir=output_dir,
            )

            if idx % 20 == 0 or idx == num_images:
                print(f"  Evaluated & exported [{idx}/{num_images}] -> {output_dir}")

    return {
        "dice": total_dice / num_images,
        "iou": total_iou / num_images,
        "precision": total_precision / num_images,
        "recall": total_recall / num_images,
        "num_images": num_images,
    }


if __name__ == '__main__':
    datasets = ["test"]
    bbox_coords_test = {}
    test_files = []
    missed_count = 0

    yolo_model = InferenceSaver(MODEL, conf=0.25, iou=0.5)
    yolo_model.load_model()

    for ds in datasets:
        image_root = f"./data/{DATA}/images/{ds}"
        images_path_list = sorted(
            f for f in os.listdir(image_root)
            if f.lower().endswith(VALID_EXTS)
        )

        for img_name in images_path_list:
            image_path = os.path.join(image_root, img_name)
            prompt_box = yolo_model.inference(image_path)

            if prompt_box is None:
                print(f"[WARNING] No YOLO detection for {img_name} ({ds}).")
                missed_count += 1
                continue

            bbox_coords_test[img_name] = prompt_box
            test_files.append(img_name)

    print(f"Total Test Images : {len(test_files)} (YOLO missed: {missed_count})")

    model_type = 'vit_b'
    checkpoint = './checkpoints/sam_vit_b_01ec64.pth'

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    test_results = test(
        model=model,
        bbox_coords_test=bbox_coords_test,
        test_files=test_files,
        output_dir=VISUALS_DIR,
        include_missed_detections=True,
    )

    print(f"\nFinal Test Results: {MODEL_PATH}")
    print(f"Visuals Saved To:   {VISUALS_DIR.resolve()}")
    print(f"Dice:      {test_results['dice']:.4f}")
    print(f"IoU:       {test_results['iou']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall:    {test_results['recall']:.4f}")