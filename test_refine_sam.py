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

NUM_CONTOUR_POINTS = 5
CONTOUR_POINT_LABEL = 1  # 1 = foreground prompt, 0 = background prompt
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

np.random.seed(RANDOM_SEED)


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


def sample_mask_contour_points(
    mask: np.ndarray,
    num_points: int = 10,
    point_label: int = 1
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Expands the binary mask outward by 3 pixels via dilation, extracts its contour, 
    and samples `num_points` evenly spaced (equidistant) points along that outer boundary path.
    Returns (point_coords, point_labels) formatted for SamPredictor.predict().
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    # Dilate mask outward by 3 pixels (7x7 ellipse kernel has a radius of 3 pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(mask_u8, kernel, iterations=1)

    # Extract contour from the dilated (outer) mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None, None

    # Select the largest contour by area to focus on the main object
    contour = max(contours, key=cv2.contourArea)
    all_pts = contour.reshape(-1, 2).astype(np.float32)  # Shape: (M, 2)
    
    total_available = len(all_pts)
    if total_available == 0:
        return None, None

    # If the contour has fewer pixels than requested points, return all available
    if total_available <= num_points:
        point_coords = all_pts
        point_labels = np.full((total_available,), point_label, dtype=np.int32)
        return point_coords, point_labels

    # Calculate segment lengths between consecutive contour points
    diffs = np.diff(all_pts, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Compute cumulative arc length along the contour path
    cum_dist = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    total_length = cum_dist[-1]

    if total_length == 0:
        point_coords = all_pts[:1]
        point_labels = np.full((1,), point_label, dtype=np.int32)
        return point_coords, point_labels

    # Generate target arc lengths spaced evenly along the perimeter
    target_dists = np.linspace(0, total_length, num_points, endpoint=False)

    # Find the contour indices closest to each target arc length
    chosen_indices = [np.argmin(np.abs(cum_dist - target)) for target in target_dists]

    point_coords = all_pts[chosen_indices].astype(np.float32)
    point_labels = np.full((num_points,), point_label, dtype=np.int32)

    return point_coords, point_labels


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


def draw_prompt_points(ax, point_coords):
    """Draws sampled 2nd-prompt contour points onto a Matplotlib axis."""
    if point_coords is None or len(point_coords) == 0:
        return
    pts = np.array(point_coords)
    ax.scatter(
        pts[:, 0], pts[:, 1],
        c="cyan", edgecolors="black", s=45, linewidths=1.0, zorder=5,
        label="Contour Pts"
    )


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-6) -> dict:
    """Computes Dice, IoU, Precision, and Recall between binary masks."""
    pred_flat = pred_mask.reshape(-1)
    gt_flat = gt_mask.reshape(-1)

    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))

    dice = (2.0 * TP + smooth) / (2.0 * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)

    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def save_eval_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask_1: np.ndarray,
    pred_mask_2: np.ndarray,
    prompt_box,
    contour_points: np.ndarray | None,
    image_name: str,
    dice_1: float,
    dice_2: float,
    iou_2: float,
    output_dir: Path,
):
    """Exports a 5-panel Matplotlib figure showing 1st mask + contour points and 2nd refined mask."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    h, w = gt_mask.shape

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(
        f"{image_name}  |  1st Dice: {dice_1:.4f} -> 2nd Dice: {dice_2:.4f} (IoU: {iou_2:.4f})",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: Raw Image + YOLO Box
    axes[0].imshow(image)
    draw_yolo_boxes(axes[0], prompt_box)
    box_status = "Detected" if prompt_box is not None else "MISSED"
    axes[0].set_title(f"1. Image + YOLO Box ({box_status})")
    axes[0].axis("off")

    # Panel 2: Ground Truth Mask
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("2. Ground Truth Mask")
    axes[1].axis("off")

    # Panel 3: 1st Mask + Sampled Contour Points
    axes[2].imshow(pred_mask_1, cmap="gray", vmin=0, vmax=1)
    draw_prompt_points(axes[2], contour_points)
    axes[2].set_title(f"3. 1st Mask + {NUM_CONTOUR_POINTS} Contour Pts")
    axes[2].axis("off")

    # Panel 4: 2nd Refined Segmentation Mask
    axes[3].imshow(pred_mask_2, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("4. 2nd Mask (Refined)")
    axes[3].axis("off")

    # Panel 5: Refined Overlay (Image + Box + Contour Pts + TP/FP/FN Map)
    axes[4].imshow(image)
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[(pred_mask_2 == 1) & (gt_mask == 1)] = [0.0, 1.0, 0.0, 0.45]  # TP: Green
    overlay[(pred_mask_2 == 1) & (gt_mask == 0)] = [1.0, 0.0, 0.0, 0.45]  # FP: Red
    overlay[(pred_mask_2 == 0) & (gt_mask == 1)] = [0.0, 0.4, 1.0, 0.45]  # FN: Blue
    axes[4].imshow(overlay)
    draw_yolo_boxes(axes[4], prompt_box)
    draw_prompt_points(axes[4], contour_points)
    axes[4].set_title("5. Refined Overlay (Grn:TP|Red:FP|Blu:FN)")
    axes[4].axis("off")

    out_path = output_dir / f"{stem}_eval.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def predict_with_contour_refinement(
    predictor: SamPredictor,
    single_box: np.ndarray,
    num_points: int = NUM_CONTOUR_POINTS,
    point_label: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Runs 1st prompt (YOLO box) -> samples 10 random points on 1st mask contour ->
    runs 2nd prompt (box + 10 contour points + 1st mask logits).
    Returns (mask_1st, mask_2nd, sampled_points).
    """
    # 1st Prompt: Box only
    pred_1, _, logits_1 = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=single_box,
        multimask_output=False,
    )
    mask_1 = (pred_1[0] > 0).astype(np.uint8)

    # Sample points along the contour of mask_1
    pt_coords, pt_labels = sample_mask_contour_points(
        mask_1, num_points=num_points, point_label=point_label
    )

    if pt_coords is None:
        return mask_1, mask_1.copy(), None

    # 2nd Prompt: Box + 10 contour points + logits_1
    pred_2, _, _ = predictor.predict(
        point_coords=pt_coords,
        point_labels=pt_labels,
        box=single_box,
        mask_input=logits_1,
        multimask_output=False,
    )
    mask_2 = (pred_2[0] > 0).astype(np.uint8)

    return mask_1, mask_2, pt_coords


def test(model, bbox_coords_test, test_files, output_dir: Path, include_missed_detections=True):
    image_root = f"./data/{DATA}/images/test"
    gt_root    = f"./data/{DATA}/masks/test"

    model.eval()
    predictor_tuned = SamPredictor(model)

    images_path_list = sorted(test_files)
    num_images = len(images_path_list)

    if num_images == 0:
        raise ValueError("No test images to evaluate!")

    totals_1st = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    totals_2nd = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    with torch.no_grad():
        for idx, image_name in enumerate(images_path_list, 1):
            img_path = os.path.join(image_root, image_name)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask_path = find_mask_path(gt_root, image_name)
            gt_mask = load_binary_mask(mask_path)
            h, w = gt_mask.shape

            prompt_box = bbox_coords_test.get(image_name, None)
            all_sampled_pts = None

            if prompt_box is None:
                if not include_missed_detections:
                    continue
                pred_mask_1 = np.zeros((h, w), dtype=np.uint8)
                pred_mask_2 = np.zeros((h, w), dtype=np.uint8)
            else:
                predictor_tuned.set_image(image)
                prompt_box_np = np.array(prompt_box)

                if prompt_box_np.ndim == 1:
                    pred_mask_1, pred_mask_2, all_sampled_pts = predict_with_contour_refinement(
                        predictor_tuned,
                        prompt_box_np,
                        num_points=NUM_CONTOUR_POINTS,
                        point_label=CONTOUR_POINT_LABEL,
                    )
                else:
                    combined_1 = np.zeros((h, w), dtype=np.uint8)
                    combined_2 = np.zeros((h, w), dtype=np.uint8)
                    pts_list = []

                    for single_box in prompt_box_np:
                        m1, m2, pts = predict_with_contour_refinement(
                            predictor_tuned,
                            single_box,
                            num_points=NUM_CONTOUR_POINTS,
                            point_label=CONTOUR_POINT_LABEL,
                        )
                        combined_1 = np.logical_or(combined_1, m1 > 0)
                        combined_2 = np.logical_or(combined_2, m2 > 0)
                        if pts is not None:
                            pts_list.append(pts)

                    pred_mask_1 = combined_1.astype(np.uint8)
                    pred_mask_2 = combined_2.astype(np.uint8)
                    if pts_list:
                        all_sampled_pts = np.vstack(pts_list)

            metrics_1 = compute_metrics(pred_mask_1, gt_mask)
            metrics_2 = compute_metrics(pred_mask_2, gt_mask)

            for k in totals_1st:
                totals_1st[k] += metrics_1[k]
                totals_2nd[k] += metrics_2[k]

            save_eval_visualization(
                image=image,
                gt_mask=gt_mask,
                pred_mask_1=pred_mask_1,
                pred_mask_2=pred_mask_2,
                prompt_box=prompt_box,
                contour_points=all_sampled_pts,
                image_name=image_name,
                dice_1=metrics_1["dice"],
                dice_2=metrics_2["dice"],
                iou_2=metrics_2["iou"],
                output_dir=output_dir,
            )

            if idx % 20 == 0 or idx == num_images:
                print(f"  Evaluated & exported [{idx}/{num_images}] -> {output_dir}")

    return {
        "1st_prompt": {k: v / num_images for k, v in totals_1st.items()},
        "2nd_prompt": {k: v / num_images for k, v in totals_2nd.items()},
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

    model = sam_model_registry[model_type]()
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

    r1 = test_results["1st_prompt"]
    r2 = test_results["2nd_prompt"]

    print(f"\nFinal Test Results: {MODEL_PATH}")
    print(f"Visuals Saved To:   {VISUALS_DIR.resolve()}")
    print("-" * 52)
    print(f"{'Metric':<12} | {'1st Prompt (Box)':<16} | {'2nd Prompt (+10 Pts)':<18}")
    print("-" * 52)
    print(f"{'Dice':<12} | {r1['dice']:<16.4f} | {r2['dice']:<18.4f}")
    print(f"{'IoU':<12} | {r1['iou']:<16.4f} | {r2['iou']:<18.4f}")
    print(f"{'Precision':<12} | {r1['precision']:<16.4f} | {r2['precision']:<18.4f}")
    print(f"{'Recall':<12} | {r1['recall']:<16.4f} | {r2['recall']:<18.4f}")
    print("-" * 52)