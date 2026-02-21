import onnxruntime as ort
import numpy as np
import cv2
import math
from PIL import Image
from pathlib import Path
import argparse
from dataclasses import dataclass

# Helper classes and functions for post-processing (portably extracted from evaluate_docquad_models.py)

@dataclass
class Letterbox:
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    offset_x: float
    offset_y: float

    @classmethod
    def create(cls, src_w: int, src_h: int, dst_w: int = 256, dst_h: int = 256) -> "Letterbox":
        scale = min(dst_w / src_w, dst_h / src_h)
        new_w = src_w * scale
        new_h = src_h * scale
        offset_x = (dst_w - new_w) / 2.0
        offset_y = (dst_h - new_h) / 2.0
        return cls(src_w, src_h, dst_w, dst_h, scale, offset_x, offset_y)

    def inverse(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale)

@dataclass
class AppPostprocessResult:
    corners_original: list[tuple[float, float]]
    source: str
    penalty_corner: float
    penalty_mask: float

def _refine_corners_64_to_256_3x3(corner_heatmaps):
    corners256 = []
    hm_all = corner_heatmaps[0]
    for c in range(4):
        hm = hm_all[c]
        idx = np.argmax(hm)
        best_y, best_x = np.unravel_index(idx, hm.shape)
        y0, y1 = max(0, best_y - 1), min(63, best_y + 1)
        x0, x1 = max(0, best_x - 1), min(63, best_x + 1)
        window = hm[y0:y1+1, x0:x1+1]
        max_logit = np.max(window)
        sum_w, sum_x, sum_y = 0.0, 0.0, 0.0
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                w = math.exp(hm[y, x] - max_logit)
                sum_w += w
                sum_x += w * (x + 0.5)
                sum_y += w * (y + 0.5)
        if sum_w == 0:
            x64, y64 = best_x + 0.5, best_y + 0.5
        else:
            x64, y64 = sum_x / sum_w, sum_y / sum_w
        corners256.append((x64 * 4.0, y64 * 4.0))
    return corners256

def _quad_from_mask_256(mask_logits):
    from scipy.stats import multivariate_normal
    mask = 1.0 / (1.0 + np.exp(-mask_logits[0, 0]))
    threshold = 0.5
    y_idx, x_idx = np.where(mask > threshold)
    if len(x_idx) < 10:
        return [(0.0, 0.0), (255.0, 0.0), (255.0, 255.0), (0.0, 255.0)]
    
    pts = np.column_stack((x_idx, y_idx)).astype(np.float32)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # PCA approximation for a rectangle/quad
    # Since we want a simple portable script, we use a simplified version of quad extraction 
    # here (bounding box of principal components).
    # For full "product" fidelity, we would need to port the exact PCA quad fitting.
    # Here, for simplicity, we use extreme points in principal directions.
    
    p1 = eigvecs[:, 1] # Main direction
    p2 = eigvecs[:, 0] # Secondary direction
    
    proj1 = centered @ p1
    proj2 = centered @ p2
    
    m1, M1 = np.min(proj1), np.max(proj1)
    m2, M2 = np.min(proj2), np.max(proj2)
    
    # Corner points in PCA space
    rect_pca = np.array([
        [m1, m2], [M1, m2], [M1, M2], [m1, M2]
    ])
    
    # Back to image space (64x64)
    rect_64 = rect_pca @ eigvecs.T + mean
    
    # Sort by TL, TR, BR, BL
    rect_64 = rect_64[np.argsort(np.arctan2(rect_64[:, 1] - mean[1], rect_64[:, 0] - mean[0]))]
    # Simple re-ordering for TL start
    tl_idx = np.argmin(np.sum(rect_64, axis=1))
    rect_64 = np.roll(rect_64, -tl_idx, axis=0)
    
    return [(float(x * 4.0), float(y * 4.0)) for x, y in rect_64]

def _shoelace_area(pts):
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - y1 * x2
    return abs(s) * 0.5

def _is_convex_quad(pts):
    signs = []
    for i in range(len(pts)):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % len(pts)]
        cx, cy = pts[(i + 2) % len(pts)]
        z = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(z) < 1e-9: continue
        signs.append(1 if z > 0 else -1)
    return len(signs) > 0 and all(s == signs[0] for s in signs)

def _apply_product_postprocessing_portable(corner_heatmaps, mask_logits, lb):
    corners_c = _refine_corners_64_to_256_3x3(corner_heatmaps)
    
    # Path selection logic (simplified for portability, but conceptually identical)
    area_c = _shoelace_area(corners_c)
    is_convex = _is_convex_quad(corners_c)
    
    # If corner quad is valid, we take it primarily (as is mostly the case in the app)
    if is_convex and area_c > 1000:
        final_corners_256 = corners_c
        source = "corners"
    else:
        # Fallback to mask (simplified)
        final_corners_256 = _quad_from_mask_256(mask_logits)
        source = "mask"
        
    corners_orig = [lb.inverse(x, y) for x, y in final_corners_256]
    return AppPostprocessResult(corners_orig, source, 0.0, 0.0)

def visualize(image_path, model_path):
    # 1. Load model
    print(f"Loading model: {model_path}")
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return

    sess_options = ort.SessionOptions()
    if str(model_path).endswith(".ort"):
        sess_options.add_session_config_entry("session.load_model_format", "ORT")
    
    session = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # 2. Load and preprocess image
    print(f"Loading image: {image_path}")
    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    orig_h, orig_w = orig_img.shape[:2]
    lb = Letterbox.create(orig_w, orig_h, 256, 256)
    
    # Letterbox preparation
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    new_w, new_h = int(orig_w * lb.scale), int(orig_h * lb.scale)
    img_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (256, 256), (0, 0, 0))
    canvas.paste(img_resized, (int(lb.offset_x), int(lb.offset_y)))
    
    # Normalization
    input_data = np.array(canvas, dtype=np.float32) / 255.0
    input_data = input_data.transpose(2, 0, 1) # HWC -> CHW
    input_data = np.expand_dims(input_data, axis=0) # NCHW
    
    # 3. Inference
    print("Starting inference...")
    outputs = session.run(None, {input_name: input_data})
    corner_heatmaps = outputs[0]
    mask_logits = outputs[1] if len(outputs) > 1 else np.zeros((1, 1, 64, 64), dtype=np.float32)
    
    # 4. Post-Processing (Portable Version)
    app_result = _apply_product_postprocessing_portable(corner_heatmaps, mask_logits, lb)
    corners_orig = app_result.corners_original
    print(f"Detection via: {app_result.source}")
    
    # 5. Visualization
    vis_img = orig_img.copy()
    pts = np.array(corners_orig, np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    colors = [(0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0)]
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(corners_orig):
        cv2.circle(vis_img, (int(x), int(y)), 10, colors[i], -1)
        cv2.putText(vis_img, labels[i], (int(x) + 10, int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
    
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, vis_img)
    print(f"Result saved at: {output_path}")
    
    # If possible, show image (only works in environments with a display)
    # cv2.imshow("Corner Detection", vis_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # Default model path relative to the script
    script_dir = Path(__file__).parent
    default_model = script_dir.parent / "model" / "docquadnet256_trained_opset17.ort"

    parser = argparse.ArgumentParser(description="DocQuadNet Corner Detection Visualization")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default=str(default_model), 
                        help="Path to the .ort or .onnx model")
    
    args = parser.parse_args()
    visualize(args.image, args.model)
