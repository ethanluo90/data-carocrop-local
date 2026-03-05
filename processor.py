"""
Carousell Image Cropping Pipeline
==================================
Crops images to Singapore Carousell marketplace standard:
- 1:1 square aspect ratio
- 1080x1080px output size
- Center-crop (no background removal)
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import pillow_heif

# Register HEIF/HEIC opener with PIL
pillow_heif.register_heif_opener()

# Carousell photo specifications
TARGET_SIZE = 1080  # 1080x1080px square
PLATFORM_X_TRIM_PCT = 0.05
PLATFORM_Y_TRIM_PCT = 0.03
DEFAULT_PADDING_PCT = 0.04
YELLOW_MIN_AREA = 350
EDGE_CLEAN_SCAN_PCT = 0.20
EDGE_CLEAN_WINDOW = 24
EDGE_NONWHITE_MAX_RATIO = 0.08
EDGE_CLEAN_STABILITY_WINDOWS = 3
EDGE_CLEAN_PRODUCT_GUARD_PCT = 0.02
ABSOLUTE_CLEAN_MARGIN_PCT = 0.08


def load_image(image_path: Path) -> Image.Image:
    """Load an image from various formats including HEIC."""
    img = Image.open(image_path)
    # Handle EXIF orientation
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    return img


def sample_backdrop_color(image: Image.Image) -> tuple:
    """
    Sample the dominant backdrop color from the image edges.
    
    Looks at the outer 15% margin of the image and finds the most common
    light color (brightness > 150) to use as the canvas background.
    
    Returns:
        tuple: RGB color (r, g, b) of the dominant backdrop color
    """
    import numpy as np
    from collections import Counter
    
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Define edge zones (outer 15%)
    margin_h = int(height * 0.15)
    margin_w = int(width * 0.15)
    
    # Collect pixels from all edge zones
    edge_pixels = []
    
    # Top strip
    edge_pixels.extend(img_array[:margin_h, :].reshape(-1, 3).tolist())
    # Bottom strip
    edge_pixels.extend(img_array[-margin_h:, :].reshape(-1, 3).tolist())
    # Left strip (excluding corners already counted)
    edge_pixels.extend(img_array[margin_h:-margin_h, :margin_w].reshape(-1, 3).tolist())
    # Right strip (excluding corners already counted)
    edge_pixels.extend(img_array[margin_h:-margin_h, -margin_w:].reshape(-1, 3).tolist())
    
    # Filter to only light pixels (potential backdrop, not dark product edges)
    edge_pixels = np.array(edge_pixels)
    brightness = np.mean(edge_pixels, axis=1)
    light_pixels = edge_pixels[brightness > 150]
    
    if len(light_pixels) == 0:
        # Fallback to white if no light pixels found
        print("     No backdrop detected, using white")
        return (255, 255, 255)
    
    # Find median color of light pixels (more robust than mean)
    median_color = np.median(light_pixels, axis=0).astype(int)
    
    print(f"     Sampled backdrop color: RGB({median_color[0]}, {median_color[1]}, {median_color[2]})")
    
    return tuple(median_color)


def brighten_backdrop(image: Image.Image, brightness_boost: float = 1.2) -> Image.Image:
    """
    Selectively brighten only the backdrop area while preserving product colors.
    
    Uses edge detection and contour finding to create a mask separating product
    from backdrop, then applies brightness boost only to backdrop pixels.
    
    Args:
        image: Input image (already cropped)
        brightness_boost: Brightness multiplier for backdrop (1.0 = no change)
    
    Returns:
        Image with brightened backdrop and original product
    """
    import numpy as np
    import cv2
    
    # Convert to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to close gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask (0 = backdrop, 255 = product)
    product_mask = np.zeros((height, width), dtype=np.uint8)
    
    if contours:
        # Fill largest contour (product) with white
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(product_mask, [largest_contour], -1, 255, -1)
    
    # Invert mask (255 = backdrop, 0 = product)
    backdrop_mask = cv2.bitwise_not(product_mask)
    
    # SOFT MASK: Apply Gaussian blur for smooth transition (eliminates visible border)
    soft_mask = cv2.GaussianBlur(backdrop_mask.astype(np.float32), (51, 51), 0)
    soft_mask = soft_mask / 255.0  # Normalize to 0-1 range
    
    # Measure backdrop brightness
    backdrop_pixels = img_array[backdrop_mask > 0]
    if len(backdrop_pixels) > 0:
        avg_backdrop_brightness = np.mean(backdrop_pixels)
        
        # Only brighten if backdrop is dark (< 200)
        if avg_backdrop_brightness < 200:
            # Apply brightness boost to entire image
            brightened = np.clip(img_array * brightness_boost, 0, 255).astype(np.uint8)
            
            # Blend using SOFT mask (smooth transition, no visible border)
            result = (brightened * soft_mask[:, :, np.newaxis] + 
                      img_array * (1 - soft_mask[:, :, np.newaxis])).astype(np.uint8)
            
            print(f"     Backdrop brightened: avg={avg_backdrop_brightness:.1f} -> boost={brightness_boost:.2f}x")
        else:
            # Backdrop already bright, no change
            result = img_array
            print(f"     Backdrop already bright: avg={avg_backdrop_brightness:.1f}, no boost needed")
    else:
        # No backdrop detected, return original
        result = img_array
        print(f"     No backdrop detected, skipping brightening")
    
    return Image.fromarray(result)


def enhance_image(image: Image.Image, adaptive: bool = True) -> Image.Image:
    """
    Enhance image with adaptive or fixed brightness, contrast, and sharpening.
    
    Args:
        image: Input image
        adaptive: If True, analyze image and adjust parameters dynamically
    
    Returns:
        Enhanced image with applied adjustments
    """
    import numpy as np
    import cv2
    
    if adaptive:
        # Convert to numpy for analysis
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # BRIGHTNESS ANALYSIS
        # Measure average brightness (0-255)
        brightness_avg = np.mean(gray)
        
        if brightness_avg < 100:           # Dark image
            brightness = 1.35              # Brighten more
        elif brightness_avg < 150:         # Medium dim image
            brightness = 1.25              # Brighten moderately
        elif brightness_avg > 200:         # Very bright image
            brightness = 1.05              # Brighten less
        else:                              # Normal brightness
            brightness = 1.15              # Standard boost
        
        # CONTRAST ANALYSIS
        # Measure contrast using standard deviation
        contrast_level = np.std(gray)
        
        if contrast_level < 35:            # Low contrast (flat/washed out)
            contrast = 1.25                # Boost more
        elif contrast_level > 65:          # High contrast
            contrast = 1.05                # Boost less
        else:                              # Normal contrast
            contrast = 1.15                # Standard boost
        
        # SHARPNESS ANALYSIS
        # Detect sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()
        
        if sharpness_score < 100:          # Blurry image
            sharpness = 1.5                # Sharpen more
        elif sharpness_score > 500:        # Already sharp
            sharpness = 1.1                # Sharpen less
        else:                              # Normal sharpness
            sharpness = 1.3                # Standard sharpen
        
        print(f"     Analysis: brightness={brightness_avg:.1f}, contrast={contrast_level:.1f}, sharpness={sharpness_score:.1f}")
    else:
        # Fixed values (original behavior)
        brightness = 1.15
        contrast = 1.05
        sharpness = 1.3
    
    # Apply enhancements
    enhancer = ImageEnhance.Brightness(image)
    enhanced = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(sharpness)
    
    print(f"     Enhanced: brightness={brightness:.2f}, contrast={contrast:.2f}, sharpness={sharpness:.2f}")
    
    return enhanced

def extend_edges(image: Image.Image, extend_percent: float = 0.15) -> Image.Image:
    """
    Extend the image edges outward by tiling the edge pixels.
    
    This creates extra margin around the image by repeating the edge pixels,
    giving more backdrop area for the smart crop to work with.
    
    Args:
        image: Input image
        extend_percent: How much to extend each edge (as percentage of image size)
    
    Returns:
        Extended image with tiled edges
    """
    import numpy as np
    
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Calculate extension size
    extend_h = int(height * extend_percent)
    extend_w = int(width * extend_percent)
    
    # Create new larger canvas
    new_height = height + 2 * extend_h
    new_width = width + 2 * extend_w
    
    # Start with tiled edges
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Place original image in center
    result[extend_h:extend_h+height, extend_w:extend_w+width] = img_array
    
    # Extend TOP edge (tile the top row)
    top_row = img_array[0:1, :, :]  # First row
    for y in range(extend_h):
        result[y, extend_w:extend_w+width] = top_row
    
    # Extend BOTTOM edge (tile the bottom row)
    bottom_row = img_array[-1:, :, :]  # Last row
    for y in range(extend_h + height, new_height):
        result[y, extend_w:extend_w+width] = bottom_row
    
    # Extend LEFT edge (tile the left column)
    left_col = img_array[:, 0:1, :]  # First column
    for x in range(extend_w):
        result[extend_h:extend_h+height, x] = left_col[:, 0, :]
    
    # Extend RIGHT edge (tile the right column)
    right_col = img_array[:, -1:, :]  # Last column
    for x in range(extend_w + width, new_width):
        result[extend_h:extend_h+height, x] = right_col[:, 0, :]
    
    # Fill corners with corner pixels
    # Top-left corner
    result[:extend_h, :extend_w] = img_array[0, 0]
    # Top-right corner
    result[:extend_h, extend_w+width:] = img_array[0, -1]
    # Bottom-left corner
    result[extend_h+height:, :extend_w] = img_array[-1, 0]
    # Bottom-right corner
    result[extend_h+height:, extend_w+width:] = img_array[-1, -1]
    
    print(f"     Extended edges by {extend_percent*100:.0f}%: {width}x{height} -> {new_width}x{new_height}")
    
    return Image.fromarray(result)


def _clip_ltrb(bounds: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    left = max(0, min(width - 1, int(left)))
    top = max(0, min(height - 1, int(top)))
    right = max(left + 1, min(width, int(right)))
    bottom = max(top + 1, min(height, int(bottom)))
    return left, top, right, bottom


def _union_ltrb(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _xywh_to_ltrb(bounds_xywh: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = bounds_xywh
    return x, y, x + w, y + h


def score_side_contamination(
    img,
    side,
    x1,
    x2,
    row_mask,
    bg_ref,
) -> float:
    """Score how likely a side window is edge clutter instead of clean backdrop."""
    import numpy as np
    import cv2

    height = img.shape[0]
    x1 = max(0, min(img.shape[1] - 1, int(x1)))
    x2 = max(x1 + 1, min(img.shape[1], int(x2)))
    window = img[:, x1:x2, :]
    if window.size == 0:
        return 0.0

    if row_mask is not None and len(row_mask) == height:
        active = bool(np.any(row_mask))
        if active:
            window = window[row_mask, :, :]
    if window.size == 0:
        return 0.0

    gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(window, cv2.COLOR_RGB2LAB)

    luma = float(gray.mean())
    sat = float((window.max(axis=2) - window.min(axis=2)).mean())
    a_mean = float(lab[:, :, 1].mean())
    b_mean = float(lab[:, :, 2].mean())

    luma_delta = abs(luma - bg_ref["luma"]) / 255.0
    sat_delta = max(0.0, sat - bg_ref["sat"]) / 255.0
    chroma_delta = min(
        1.0,
        (((a_mean - bg_ref["a"]) ** 2 + (b_mean - bg_ref["b"]) ** 2) ** 0.5) / 181.0,
    )

    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(edges.mean() / 255.0)
    texture_var = min(1.0, float(gray.var()) / 500.0)

    score = (
        0.30 * luma_delta
        + 0.15 * sat_delta
        + 0.20 * chroma_delta
        + 0.20 * edge_density
        + 0.15 * texture_var
    )
    return float(max(0.0, min(1.0, score)))


def filter_ai_components(
    mask,
    image: Image.Image,
    platform_hint,
    required_center_hint,
):
    """Drop side-touching AI contours that look like backdrop artifacts.
    
    Uses center-biased scoring: score = area * (1 - dist/max_dist)^2
    with 0.3x penalty for edge-touching contours.
    """
    import numpy as np
    import cv2

    img = np.array(image.convert("RGB"))
    height, width = img.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    if required_center_hint is None:
        center_x, center_y = width // 2, height // 2
    else:
        center_x, center_y = required_center_hint

    if platform_hint is None:
        plat_left, plat_top, plat_right, plat_bottom = 0, 0, width, height
    else:
        plat_left, plat_top, plat_right, plat_bottom = _clip_ltrb(platform_hint, width, height)

    cy1 = max(plat_top, int(height * 0.15))
    cy2 = min(plat_bottom, int(height * 0.90))
    cx1 = max(plat_left, int(width * 0.35))
    cx2 = min(plat_right, int(width * 0.65))
    bg = img[cy1:cy2, cx1:cx2, :]
    if bg.size == 0:
        bg = img[int(height * 0.25):int(height * 0.85), int(width * 0.35):int(width * 0.65), :]
    if bg.size == 0:
        bg = img

    bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB)
    bg_ref = {
        "luma": float(bg_gray.mean()),
        "sat": float((bg.max(axis=2) - bg.min(axis=2)).mean()),
        "a": float(bg_lab[:, :, 1].mean()),
        "b": float(bg_lab[:, :, 2].mean()),
    }

    min_area = max(200.0, width * height * 0.0005)
    # Border margins: 5% extreme auto-drop, 8% side scoring zone
    extreme_edge_x = int(width * 0.05)
    extreme_edge_y = int(height * 0.05)
    side_margin = int(width * 0.08)
    top_margin = int(height * 0.05)
    bottom_margin = int(height * 0.03)
    max_dist = ((width / 2) ** 2 + (height / 2) ** 2) ** 0.5

    components = []
    kept = 0
    dropped = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + (w // 2), y + (h // 2)
        dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        fill_ratio = area / max(1.0, float(w * h))
        aspect = max(w / max(1.0, h), h / max(1.0, w))

        # Edge touching checks (left/right AND top/bottom)
        touch_left = x <= side_margin
        touch_right = x + w >= width - side_margin
        touch_top = y <= top_margin
        touch_bottom = y + h >= height - bottom_margin
        side_touch = touch_left or touch_right

        row_mask = np.zeros((height,), dtype=bool)
        y1 = max(0, y - int(h * 0.25))
        y2 = min(height, y + h + int(h * 0.25))
        if y2 > y1:
            row_mask[y1:y2] = True

        contam = 0.0
        if side_touch:
            contam = score_side_contamination(img, "left" if touch_left else "right", x, x + w, row_mask, bg_ref)

        # --- CENTER-BIASED COMPOSITE SCORE ---
        # score = area * (1 - dist/max_dist)^2, penalize edge-touching
        proximity = max(0.0, 1.0 - dist / max_dist)
        score = area * (proximity ** 2)
        
        # Penalize edge-touching contours
        any_edge_touch = touch_left or touch_right or touch_top or touch_bottom
        if any_edge_touch:
            score *= 0.3

        # --- DROP LOGIC ---
        # 1. Auto-drop: contours touching extreme outer 5% of any edge
        is_extreme_edge = (
            (x <= extreme_edge_x) or
            (x + w >= width - extreme_edge_x) or
            (y <= extreme_edge_y) or
            (y + h >= height - extreme_edge_y)
        )

        shape_bad = (aspect > 5.5 and fill_ratio < 0.25) or (fill_ratio < 0.12)
        smallish = area < (width * height * 0.03)
        far_from_center = dist > (min(width, height) * 0.32)
        
        should_drop = (
            is_extreme_edge or
            (side_touch and smallish and contam > 0.26 and (shape_bad or far_from_center))
        )

        if should_drop:
            dropped += 1
            print(
                f"     [AI-COMPONENT] dropped artifact at ({x},{y}) "
                f"{w}x{h}, score={score:.0f}, contam={contam:.3f}, fill={fill_ratio:.3f}"
            )
            continue

        kept += 1
        components.append((x, y, w, h, area, dist, score))

    print(f"     [AI-COMPONENT] kept={kept} dropped={dropped}")
    return components


def get_ai_crop_bounds(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    AI-first detection using rembg/U2-Net.
    Returns bounds as (x, y, w, h).
    """
    try:
        from rembg import remove
        import numpy as np
        import cv2

        result = remove(image, alpha_matting=False)
        alpha = np.array(result)[:, :, 3]
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print('     [AI-FIRST] No contours from AI alpha mask')
            return None

        img_h, img_w = alpha.shape
        center_x, center_y = img_w // 2, img_h // 2
        components = filter_ai_components(
            mask=mask,
            image=image,
            platform_hint=None,
            required_center_hint=(center_x, center_y),
        )

        # Pick seed by highest composite score (center-biased, area-weighted).
        seed = max(components, key=lambda c: c[6])
        seed_cx = seed[0] + seed[2] // 2
        seed_cy = seed[1] + seed[3] // 2
        max_cluster_dist = min(img_w, img_h) * 0.35
        seed_area = seed[4]
        total_img_area = img_w * img_h

        cluster = []
        for c in components:
            cx = c[0] + c[2] // 2
            cy = c[1] + c[3] // 2
            dist = ((cx - seed_cx) ** 2 + (cy - seed_cy) ** 2) ** 0.5
            c_area = c[4]
            
            # MULTI-PART COMPONENT FILTER
            # Keep if it is close to the seed AND/OR if it is a significant component
            # (at least 0.5% size of seed OR >1.0% of total image area)
            is_significant = (c_area >= seed_area * 0.005) or (c_area > total_img_area * 0.01)
            
            if dist <= max_cluster_dist or is_significant:
                cluster.append(c)

        if not cluster:
            cluster = [seed]

        left = min(c[0] for c in cluster)
        top = min(c[1] for c in cluster)
        right = max(c[0] + c[2] for c in cluster)
        bottom = max(c[1] + c[3] for c in cluster)
        print(f'     [AI-FIRST] U2-Net bounds {right-left}x{bottom-top} at ({left},{top})')
        return left, top, right - left, bottom - top

    except Exception as e:
        print(f'     [AI-FIRST] AI detection failed: {e}')
        return None


def get_cv_crop_bounds(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Fallback CV detector using Lab-based foreground masking.
    
    Converts to Lab, thresholds on L channel using Otsu,
    applies morphology cleanup, then scores contours by
    center proximity and area.
    """
    import numpy as np
    import cv2

    img = np.array(image.convert('RGB'))
    height, width = img.shape[:2]

    # Lab-based foreground mask (more stable than Canny on light backgrounds)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    blurred_l = cv2.GaussianBlur(l_channel, (5, 5), 0)
    
    # Otsu threshold on L channel (separates dark product from bright backdrop)
    _, mask = cv2.threshold(blurred_l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphology: close (fill gaps in product), then open (remove noise specks)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print('     [AI-FIRST] CV fallback found no contours')
        return None

    center_x = width // 2
    center_y = height // 2
    # Wider edge margins matching the light-box strategy
    edge_margin_x = int(width * 0.08)
    edge_margin_y = int(height * 0.05)
    min_area = (width * height) * 0.001
    max_dist = ((width / 2) ** 2 + (height / 2) ** 2) ** 0.5

    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < min_area or w < 30 or h < 30:
            continue

        # Auto-drop contours touching outer edge margins
        touching_edge = (
            x <= edge_margin_x
            or y <= edge_margin_y
            or x + w >= width - edge_margin_x
            or y + h >= height - edge_margin_y
        )
        if touching_edge:
            continue

        cx = x + w // 2
        cy = y + h // 2
        dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        
        # Center-biased composite score
        proximity = max(0.0, 1.0 - dist / max_dist)
        score = area * (proximity ** 2)
        valid_contours.append((x, y, w, h, area, dist, score))

    if not valid_contours:
        print('     [AI-FIRST] CV fallback had only edge/noise contours')
        return None

    # Sort by composite score (highest = most central + largest)
    valid_contours.sort(key=lambda c: c[6], reverse=True)
    main = valid_contours[0]
    cluster_center_x = main[0] + main[2] // 2
    cluster_center_y = main[1] + main[3] // 2
    max_cluster_dist = min(width, height) * 0.35

    cluster = [main]
    for c in valid_contours[1:]:
        cx = c[0] + c[2] // 2
        cy = c[1] + c[3] // 2
        dist = ((cx - cluster_center_x) ** 2 + (cy - cluster_center_y) ** 2) ** 0.5
        if dist <= max_cluster_dist:
            cluster.append(c)
            cluster_center_x = (cluster_center_x + cx) // 2
            cluster_center_y = (cluster_center_y + cy) // 2

    left = min(c[0] for c in cluster)
    top = min(c[1] for c in cluster)
    right = max(c[0] + c[2] for c in cluster)
    bottom = max(c[1] + c[3] for c in cluster)
    print(f'     [AI-FIRST] CV fallback bounds {right-left}x{bottom-top} at ({left},{top})')
    return left, top, right - left, bottom - top


def detect_white_platform_bounds(
    image: Image.Image,
    required_bounds: Optional[Tuple[int, int, int, int]] = None,
    strict_mode: bool = False,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect usable white platform bounds (excluding gray side edges).
    Returns (left, top, right, bottom).
    """
    import numpy as np
    import cv2

    img = np.array(image.convert('RGB'))
    height, width = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    white_l_min = 182 if strict_mode else 175
    white_chroma_max = 16 if strict_mode else 18
    white_sat_max = 58 if strict_mode else 65
    bright_v_min = 228 if strict_mode else 220
    bright_sat_max = 34 if strict_mode else 40

    neutral_white = (
        (l_channel >= white_l_min)
        & (np.abs(a_channel.astype(np.int16) - 128) <= white_chroma_max)
        & (np.abs(b_channel.astype(np.int16) - 128) <= white_chroma_max)
        & (s_channel <= white_sat_max)
    )
    bright_white = (v_channel >= bright_v_min) & (s_channel <= bright_sat_max)
    white_mask = np.where(neutral_white | bright_white, 255, 0).astype(np.uint8)

    close_k = np.ones((9, 9), np.uint8)
    open_k = np.ones((7, 7), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, close_k)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, open_k)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    if contours:
        min_area = width * height * 0.08
        target_x = width // 2
        target_y = int(height * 0.68)
        best_score = float('-inf')

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = x + (w // 2)
            cy = y + (h // 2)
            dist = ((cx - target_x) ** 2 + (cy - target_y) ** 2) ** 0.5
            score = area - (dist * 120.0)
            if score > best_score:
                best_score = score
                best_bbox = (x, y, x + w, y + h)

    # Fallback to full image if no stable white platform contour.
    if best_bbox is None:
        best_bbox = (0, 0, width, height)

    x_trim = max(int(width * PLATFORM_X_TRIM_PCT), int(width * ABSOLUTE_CLEAN_MARGIN_PCT))
    y_trim = int(height * PLATFORM_Y_TRIM_PCT)
    left = max(best_bbox[0], x_trim)
    top = max(best_bbox[1], y_trim)
    right = min(best_bbox[2], width - x_trim)
    bottom = min(best_bbox[3], height - y_trim)

    row_start = top + int((bottom - top) * 0.10)
    row_end = bottom - int((bottom - top) * 0.08)
    row_start = max(top, min(row_start, bottom - 1))
    row_end = max(row_start + 1, min(row_end, bottom))

    row_mask = np.ones((height,), dtype=bool)
    row_mask[:row_start] = False
    row_mask[row_end:] = False

    if required_bounds is not None:
        req_left, req_top, req_right, req_bottom = _clip_ltrb(required_bounds, width, height)
        y_guard = int(height * 0.06)
        block_top = max(row_start, req_top - y_guard)
        block_bottom = min(row_end, req_bottom + y_guard)
        if block_bottom > block_top:
            row_mask[block_top:block_bottom] = False
            # If product occupies most of scan rows, fallback to scan rows.
            if row_mask[row_start:row_end].sum() < max(24, int((row_end - row_start) * 0.2)):
                row_mask[row_start:row_end] = True
    else:
        req_left, req_top, req_right, req_bottom = left, top, right, bottom

    # Direct border component removal: catches small paper strips and side-seam blobs
    # that may be too localized for window-average scoring.
    artifact_map = (
        (v_channel < 210)
        | (s_channel > 48)
        | (np.abs(a_channel.astype(np.int16) - 128) > 22)
        | (np.abs(b_channel.astype(np.int16) - 128) > 22)
    ).astype(np.uint8)
    artifact_map[:row_start, :] = 0
    artifact_map[row_end:, :] = 0
    if row_mask is not None and len(row_mask) == height:
        artifact_map[~row_mask, :] = 0

    min_comp_area = max(40, int(width * height * 0.00015))
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(artifact_map, connectivity=8)
    border_pad = max(2, int(width * 0.003))
    pre_left, pre_right = left, right
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_comp_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        x2 = x + w
        y2 = y + h

        overlaps_required = not (x2 < req_left or x > req_right or y2 < req_top or y > req_bottom)
        if overlaps_required and area > int(width * height * 0.001):
            continue

        if x <= left + border_pad and w <= int(width * 0.18):
            left = max(left, min(right - 100, x2 + border_pad))
        if x2 >= right - border_pad and w <= int(width * 0.18):
            right = min(right, max(left + 100, x - border_pad))

    # Safety rollback: if component trim collapses width too hard, ignore it.
    min_platform_width = int(width * 0.35)
    if right - left < min_platform_width:
        left, right = pre_left, pre_right
        print("     [PLATFORM-EDGE] component-trim rollback (excessive shrink)")

    # Build center-background reference for side contamination scoring.
    cx1 = max(left, int(width * 0.35))
    cx2 = min(right, int(width * 0.65))
    bg_region = img[:, cx1:cx2, :]
    if bg_region.size == 0:
        bg_region = img[:, left:right, :]
    if bg_region.size == 0:
        bg_region = img

    if row_mask is not None and len(row_mask) == height and np.any(row_mask):
        bg_region = bg_region[row_mask, :, :]
    if bg_region.size == 0:
        bg_region = img

    bg_gray = cv2.cvtColor(bg_region, cv2.COLOR_RGB2GRAY)
    bg_lab = cv2.cvtColor(bg_region, cv2.COLOR_RGB2LAB)
    bg_ref = {
        "luma": float(bg_gray.mean()),
        "sat": float((bg_region.max(axis=2) - bg_region.min(axis=2)).mean()),
        "a": float(bg_lab[:, :, 1].mean()),
        "b": float(bg_lab[:, :, 2].mean()),
    }

    max_scan = int(width * EDGE_CLEAN_SCAN_PCT * (1.25 if strict_mode else 1.0))
    step = max(4, EDGE_CLEAN_WINDOW // 3)
    base_threshold = 0.14 if strict_mode else 0.16
    threshold = max(base_threshold, EDGE_NONWHITE_MAX_RATIO)
    stability = EDGE_CLEAN_STABILITY_WINDOWS

    left_start = left
    right_start = right

    left_max_allowed = right - 100
    right_min_allowed = left + 100

    left_scan_stop = min(right - EDGE_CLEAN_WINDOW, left + max_scan, left_max_allowed)
    right_scan_start = max(left + EDGE_CLEAN_WINDOW, right - max_scan, right_min_allowed)

    left_scores = []
    right_scores = []

    dirty_end = left
    clean_streak = 0
    x = left
    while x < left_scan_stop:
        x2 = min(right, x + EDGE_CLEAN_WINDOW)
        score = score_side_contamination(img, "left", x, x2, row_mask, bg_ref)
        left_scores.append(score)
        if score > threshold:
            dirty_end = max(dirty_end, x2)
            clean_streak = 0
        else:
            clean_streak += 1
            if dirty_end > left and clean_streak >= stability:
                break
        x += step

    dirty_start = right
    clean_streak = 0
    x = right
    while x > right_scan_start:
        x1 = max(left, x - EDGE_CLEAN_WINDOW)
        score = score_side_contamination(img, "right", x1, x, row_mask, bg_ref)
        right_scores.append(score)
        if score > threshold:
            dirty_start = min(dirty_start, x1)
            clean_streak = 0
        else:
            clean_streak += 1
            if dirty_start < right and clean_streak >= stability:
                break
        x -= step

    left = min(dirty_end, left_max_allowed)
    right = max(dirty_start, right_min_allowed)
    left_trim = max(0, left - left_start)
    right_trim = max(0, right_start - right)

    if right - left < 100 or bottom - top < 100:
        left, top, right, bottom = x_trim, y_trim, width - x_trim, height - y_trim

    print(
        f'     [PLATFORM] White bounds ({left},{top}) to ({right},{bottom}) '
        f'| edge-trim L={left_trim}px R={right_trim}px'
    )
    if left_scores or right_scores:
        lmax = max(left_scores) if left_scores else 0.0
        rmax = max(right_scores) if right_scores else 0.0
        print(
            f"     [PLATFORM-EDGE] threshold={threshold:.3f} strict={strict_mode} "
            f"Lmax={lmax:.3f} Rmax={rmax:.3f} "
            f"comp_trim_L={max(0, left-pre_left)} comp_trim_R={max(0, pre_right-right)}"
        )
    return left, top, right, bottom


def detect_yellow_logo_bounds(
    image: Image.Image,
    product_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect yellow mascot/logo and return union bounds (left, top, right, bottom).
    Inclusion is compulsory when detected.
    """
    import numpy as np
    import cv2

    img = np.array(image.convert('RGB'))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([15, 90, 90], dtype=np.uint8)
    upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape[:2]
    max_area = width * height * 0.02
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < YELLOW_MIN_AREA or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(1, h)
        if not (0.3 <= aspect <= 3.0):
            continue

        if product_bounds is not None:
            p_left, p_top, p_right, p_bottom = product_bounds
            cx = x + (w // 2)
            cy = y + (h // 2)
            if p_left <= cx <= p_right and p_top <= cy <= p_bottom:
                # Ignore yellow regions that are clearly on the main product body/screen.
                continue

        valid.append((x, y, x + w, y + h))

    if not valid:
        print('     [LOGO] No yellow mascot/logo detected')
        return None

    left = min(b[0] for b in valid)
    top = min(b[1] for b in valid)
    right = max(b[2] for b in valid)
    bottom = max(b[3] for b in valid)
    print(f'     [LOGO] Included yellow bounds {right-left}x{bottom-top} at ({left},{top})')
    return left, top, right, bottom


def solve_square_crop(
    required_bounds: Tuple[int, int, int, int],
    platform_bounds: Optional[Tuple[int, int, int, int]],
    image_size: Tuple[int, int],
    padding_percent: float,
    source_image: Optional[Image.Image] = None,
) -> Tuple[int, int, int, int]:
    """
    Priority solver:
    1) Keep full required bounds in frame.
    2) Use smallest valid square (max fill).
    """
    width, height = image_size
    req_left, req_top, req_right, req_bottom = _clip_ltrb(required_bounds, width, height)

    req_w = req_right - req_left
    req_h = req_bottom - req_top
    pad_x = int(req_w * padding_percent)
    pad_y = int(req_h * padding_percent)
    req_left, req_top, req_right, req_bottom = _clip_ltrb(
        (req_left - pad_x, req_top - pad_y, req_right + pad_x, req_bottom + pad_y),
        width,
        height,
    )


    square_size = max(req_right - req_left, req_bottom - req_top)
    if square_size < 2:
        square_size = min(width, height)

    # GLOBAL MINIMUM SQUARE FLOOR: ensure consistent zoom regardless of detection size.
    # When U2-Net misses low-contrast parts (e.g. white band on white background),
    # the detected bounds are too small. This floor prevents over-zoom.
    min_square = int(min(width, height) * 0.50)
    if square_size < min_square:
        print(f"     [CROP-SOLVER] Boosting square {square_size}px → {min_square}px (50% floor)")
        square_size = min_square

    if platform_bounds is None:
        platform_bounds = (0, 0, width, height)
    plat_left, plat_top, plat_right, plat_bottom = _clip_ltrb(platform_bounds, width, height)

    def pick_pos(req_min: int, req_max: int, plat_min: int, plat_max: int, size: int) -> Optional[int]:
        low = max(plat_min, req_max - size)
        high = min(plat_max - size, req_min)
        if low > high:
            return None
        ideal = (req_min + req_max - size) // 2
        return int(max(low, min(high, ideal)))

    def solve_in_bounds(bounds: Tuple[int, int, int, int], size: int) -> Optional[Tuple[int, int, int, int]]:
        b_left, b_top, b_right, b_bottom = bounds
        if size > (b_right - b_left) or size > (b_bottom - b_top):
            return None
        x = pick_pos(req_left, req_right, b_left, b_right, size)
        y = pick_pos(req_top, req_bottom, b_top, b_bottom, size)
        if x is None or y is None:
            return None
        return x, y, x + size, y + size

    crop = solve_in_bounds((plat_left, plat_top, plat_right, plat_bottom), square_size)
    if crop is not None:
        print(
            f'     [CROP-SOLVER] Platform-constrained square {square_size} '
            f'at ({crop[0]},{crop[1]})'
        )
        return crop

    # If padded bounds cannot fit the clean platform, retry with zero padding
    # before falling back to full-image limits.
    base_left, base_top, base_right, base_bottom = _clip_ltrb(required_bounds, width, height)
    base_square = max(base_right - base_left, base_bottom - base_top)
    if base_square > 1 and base_square <= (plat_right - plat_left) and base_square <= (plat_bottom - plat_top):
        prev_req = (req_left, req_top, req_right, req_bottom)
        req_left, req_top, req_right, req_bottom = base_left, base_top, base_right, base_bottom
        crop_no_pad = solve_in_bounds((plat_left, plat_top, plat_right, plat_bottom), base_square)
        req_left, req_top, req_right, req_bottom = prev_req
        if crop_no_pad is not None:
            print(
                f'     [CROP-SOLVER] Padding relaxed to preserve clean platform. '
                f'Square {base_square} at ({crop_no_pad[0]},{crop_no_pad[1]})'
            )
            return crop_no_pad

    if source_image is not None:
        strict_platform = detect_white_platform_bounds(
            source_image,
            required_bounds=(req_left, req_top, req_right, req_bottom),
            strict_mode=True,
        )
        if strict_platform is not None:
            s_left, s_top, s_right, s_bottom = _clip_ltrb(strict_platform, width, height)
            strict_crop = solve_in_bounds((s_left, s_top, s_right, s_bottom), square_size)
            if strict_crop is not None:
                print(
                    f"     [CROP-SOLVER] Strict platform retry accepted square {square_size} "
                    f"at ({strict_crop[0]},{strict_crop[1]})"
                )
                return strict_crop

    print(
        '     [CONSTRAINT-CONFLICT] Required bounds do not fully fit white platform. '
        'Falling back to full-image constraints to preserve product.'
    )
    
    # Priority 1: Full product in frame (never cut off if geometrically possible).
    # Center the square on the required bounds' center.
    base_left, base_top, base_right, base_bottom = _clip_ltrb(required_bounds, width, height)
    req_cx = (base_left + base_right) // 2
    req_cy = (base_top + base_bottom) // 2
    
    # Proportional shrink: cap to platform's smallest dimension
    # to prevent gray edges, but use 85% relative cap for consistency.
    plat_w = plat_right - plat_left
    plat_h = plat_bottom - plat_top
    max_legal_square = max(min(plat_w, plat_h), int(square_size * 0.85))
    
    if square_size > max_legal_square and max_legal_square > 0:
        print(f"     [CROP-SOLVER] Shrinking {square_size}px → {max_legal_square}px (85% relative cap)")
        square_size = max_legal_square
    
    half = square_size // 2
    crop_left = req_cx - half
    crop_top = req_cy - half
    crop_right = crop_left + square_size
    crop_bottom = crop_top + square_size
    
    # Shift if hitting image boundaries to keep crop inside the image
    if crop_left < 0:
        shift = -crop_left
        crop_left = 0
        crop_right = min(width, crop_right + shift)
    if crop_right > width:
        shift = crop_right - width
        crop_right = width
        crop_left = max(0, crop_left - shift)
        
    if crop_top < 0:
        shift = -crop_top
        crop_top = 0
        crop_bottom = min(height, crop_bottom + shift)
    if crop_bottom > height:
        shift = crop_bottom - height
        crop_bottom = height
        crop_top = max(0, crop_top - shift)
        
    print(f"     [CROP-SOLVER] Full-image fallback square {square_size} at ({crop_left},{crop_top})")
    return crop_left, crop_top, crop_right, crop_bottom


def border_contamination(strip, bg_ref=(255, 255, 255), dark_thresh=40):
    """Score a border strip for non-white contamination (0.0 = clean, 1.0 = dirty)."""
    import numpy as np
    arr = np.asarray(strip, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    diff = np.abs(arr - np.array(bg_ref, dtype=np.float32))
    max_diff = np.max(diff, axis=2)
    dirty = max_diff > dark_thresh
    return float(np.mean(dirty))


def cleanup_crop_borders(cropped: Image.Image, threshold: float = 0.15) -> Image.Image:
    """Post-crop border cleanup: scan 4 border strips + 4 corners for contamination and trim."""
    import numpy as np
    w, h = cropped.size
    if w < 100 or h < 100:
        return cropped

    img = np.array(cropped.convert('RGB'))
    strip_w = max(4, int(w * 0.08))
    strip_h = max(4, int(h * 0.08))

    left_strip = img[:, :strip_w, :]
    right_strip = img[:, w-strip_w:, :]
    top_strip = img[:strip_h, :, :]
    bottom_strip = img[h-strip_h:, :, :]

    l_score = border_contamination(left_strip)
    r_score = border_contamination(right_strip)
    t_score = border_contamination(top_strip)
    b_score = border_contamination(bottom_strip)

    trim_left = strip_w if l_score > threshold else 0
    trim_right = strip_w if r_score > threshold else 0
    trim_top = strip_h if t_score > threshold else 0
    trim_bottom = strip_h if b_score > threshold else 0

    # Corner scan: 4 quadrants scored independently
    corner_w = max(4, int(w * 0.12))
    corner_h = max(4, int(h * 0.12))
    tl_corner = img[:corner_h, :corner_w, :]
    tr_corner = img[:corner_h, w-corner_w:, :]
    bl_corner = img[h-corner_h:, :corner_w, :]
    br_corner = img[h-corner_h:, w-corner_w:, :]

    tl_score = border_contamination(tl_corner)
    tr_score = border_contamination(tr_corner)
    bl_score = border_contamination(bl_corner)
    br_score = border_contamination(br_corner)

    corner_trim_left = corner_w if (tl_score > threshold or bl_score > threshold) else 0
    corner_trim_right = corner_w if (tr_score > threshold or br_score > threshold) else 0
    corner_trim_top = corner_h if (tl_score > threshold or tr_score > threshold) else 0
    corner_trim_bottom = corner_h if (bl_score > threshold or br_score > threshold) else 0

    trim_left = max(trim_left, corner_trim_left)
    trim_right = max(trim_right, corner_trim_right)
    trim_top = max(trim_top, corner_trim_top)
    trim_bottom = max(trim_bottom, corner_trim_bottom)

    if trim_left == 0 and trim_right == 0 and trim_top == 0 and trim_bottom == 0:
        return cropped

    total_trim = max(trim_left, trim_right, trim_top, trim_bottom)
    max_trim = int(min(w, h) * 0.06)
    total_trim = min(total_trim, max_trim)

    new_left = total_trim
    new_top = total_trim
    new_right = w - total_trim
    new_bottom = h - total_trim

    if new_right - new_left < int(w * 0.80) or new_bottom - new_top < int(h * 0.80):
        print(f"     [BORDER-CLEANUP] Skipped (trim too aggressive: {total_trim}px)")
        return cropped

    print(
        f"     [BORDER-CLEANUP] Trimming {total_trim}px symmetrically "
        f"(scores L={l_score:.2f} R={r_score:.2f} T={t_score:.2f} B={b_score:.2f}, "
        f"corners TL={tl_score:.2f} TR={tr_score:.2f} BL={bl_score:.2f} BR={br_score:.2f})"
    )
    return cropped.crop((new_left, new_top, new_right, new_bottom))


def tight_crop_to_object(image: Image.Image, padding_percent: float = DEFAULT_PADDING_PCT) -> Image.Image:
    """
    AI-first crop with white-platform constraints and compulsory logo inclusion.
    Returns cropped (not yet resized) image.
    """
    width, height = image.size
    if image.mode != 'RGB':
        image = image.convert('RGB')

    ai_bounds = get_ai_crop_bounds(image)
    product_ltrb: Optional[Tuple[int, int, int, int]] = None

    if ai_bounds is not None:
        product_ltrb = _xywh_to_ltrb(ai_bounds)
    else:
        cv_bounds = get_cv_crop_bounds(image)
        if cv_bounds is not None:
            product_ltrb = _xywh_to_ltrb(cv_bounds)

    if product_ltrb is None:
        # Safety fallback: centered middle box.
        product_ltrb = (
            int(width * 0.35),
            int(height * 0.35),
            int(width * 0.65),
            int(height * 0.65),
        )
        print('     [AI-FIRST] Detector fallback to center box')

    product_ltrb = _clip_ltrb(product_ltrb, width, height)
    logo_ltrb = detect_yellow_logo_bounds(image, product_bounds=product_ltrb)
    required_ltrb = product_ltrb if logo_ltrb is None else _union_ltrb(product_ltrb, logo_ltrb)
    platform_ltrb = detect_white_platform_bounds(image, required_bounds=required_ltrb)

    crop_left, crop_top, crop_right, crop_bottom = solve_square_crop(
        required_bounds=required_ltrb,
        platform_bounds=platform_ltrb,
        image_size=(width, height),
        padding_percent=padding_percent,
        source_image=image,
    )

    square_size = crop_right - crop_left
    print(
        f'     [CROP-SOLVER] Final crop ({crop_left},{crop_top}) to '
        f'({crop_right},{crop_bottom}), size={square_size}x{square_size}'
    )
    cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return cleanup_crop_borders(cropped)


def resize_to_target(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    """Resize image to target dimensions (1080x1080)."""
    if image.size == (target_size, target_size):
        return image
    return image.resize((target_size, target_size), Image.Resampling.LANCZOS)


def apply_white_background(image: Image.Image) -> Image.Image:
    """Apply white background to image, replacing any transparency or other backgrounds."""
    # Create white background
    white_bg = Image.new('RGB', image.size, (255, 255, 255))
    
    if image.mode == 'RGBA':
        # Paste image onto white background using alpha as mask
        white_bg.paste(image, mask=image.split()[3])
    elif image.mode == 'RGB':
        white_bg = image
    else:
        # Convert other modes to RGB
        white_bg = image.convert('RGB')
    
    return white_bg


def process_image(input_path: Path, output_dir: Path) -> tuple:
    """Process a single image through the cropping pipeline."""
    print(f"\n{'='*50}")
    print(f"Processing: {input_path.name}")
    print('='*50)
    
    start_time = time.time()
    
    try:
        # Step 1: Load
        print("  -> Loading image...")
        original = load_image(input_path)
        print(f"     Original: {original.width}x{original.height}")
        
        # Step 2: Tight crop to object
        print("  -> Tight cropping to object...")
        cropped = tight_crop_to_object(original, padding_percent=0.04)
        print(f"     Output: {cropped.width}x{cropped.height}")
        
        # Step 3: Brighten backdrop (if dark)
        print("  -> Brightening backdrop...")
        backdrop_brightened = brighten_backdrop(cropped, brightness_boost=1.2)
        
        # Step 4: Enhance image (adaptive sharpening and contrast)
        print("  -> Enhancing image...")
        enhanced = enhance_image(backdrop_brightened)
        
        # Step 5: Resize to target output
        print("  -> Resizing to 1080x1080...")
        final = resize_to_target(enhanced, target_size=TARGET_SIZE)

        # Step 6: Save
        output_filename = f"{input_path.stem}_caro.png"
        output_path = output_dir / output_filename
        
        print(f"  -> Saving: {output_path.name}")
        final.save(output_path, 'PNG', quality=95)
        
        elapsed = time.time() - start_time
        print(f"  [OK] Complete! ({elapsed:.1f}s)")
        return (input_path, output_path, True)
        
    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return (input_path, None, False)


def generate_comparison(input_path: Path, output_path: Path, comparison_dir: Path) -> Optional[Path]:
    """Generate a side-by-side comparison image (input left, output right)."""
    try:
        original = load_image(input_path)
        processed = Image.open(output_path).convert('RGB')

        target_h = 1080
        orig_scale = target_h / original.height
        orig_w = int(original.width * orig_scale)
        original_resized = original.resize((orig_w, target_h), Image.LANCZOS)
        proc_resized = processed.resize((target_h, target_h), Image.LANCZOS)

        gap = 20
        label_h = 40
        canvas_w = orig_w + gap + target_h
        canvas_h = target_h + label_h
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))

        canvas.paste(original_resized, (0, label_h))
        canvas.paste(proc_resized, (orig_w + gap, label_h))

        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        draw.text((orig_w // 2 - 30, 8), "INPUT", fill=(100, 100, 100), font=font)
        draw.text((orig_w + gap + target_h // 2 - 40, 8), "OUTPUT", fill=(0, 128, 0), font=font)
        draw.line([(orig_w + gap // 2, label_h), (orig_w + gap // 2, canvas_h)], fill=(200, 200, 200), width=2)

        comp_name = f"{input_path.stem}_comparison.png"
        comp_path = comparison_dir / comp_name
        canvas.save(comp_path, 'PNG')
        return comp_path
    except Exception as e:
        print(f"  [COMPARE-ERROR] {input_path.name}: {e}")
        return None


def generate_all_comparisons(results: list, comparison_dir: Path):
    """Generate side-by-side comparisons for all successful results."""
    comparison_dir.mkdir(parents=True, exist_ok=True)
    successful = [(inp, out) for inp, out, ok in results if ok and out is not None]

    if not successful:
        print("\n[COMPARE] No successful results to compare.")
        return

    print(f"\n{'='*60}")
    print(f"  GENERATING COMPARISONS ({len(successful)} images)")
    print(f"{'='*60}")

    for idx, (inp, out) in enumerate(successful, 1):
        comp = generate_comparison(inp, out, comparison_dir)
        if comp:
            print(f"  [{idx}/{len(successful)}] {comp.name}")

    print(f"\n  [DIR] Comparisons: {comparison_dir}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Carousell Image Cropper')
    parser.add_argument('--input', type=Path, help='Input directory')
    parser.add_argument('--output', type=Path, help='Output directory')
    parser.add_argument('--compare', action='store_true', help='Generate side-by-side comparison images')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CAROUSELL IMAGE CROPPER - BATCH DRIVE MODE")
    print("  Recursive Scan & Structure Mirroring")
    print("="*60)
    
    # Setup paths
    script_dir = Path(__file__).parent
    input_dir = args.input if args.input else script_dir / "input"
    output_dir = args.output if args.output else script_dir / "output"
    
    # Create directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[DIR] Input:  {input_dir}")
    print(f"[DIR] Output: {output_dir}")
    
    # Find images (Recursive)
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'}
    print("[SCAN] Scanning for images recursively...")
    image_files = [f for f in input_dir.rglob('*') 
                   if f.is_file() and f.suffix.lower() in supported_formats]
    
    if not image_files:
        print("\n[!] No images found in input folder.")
        print("    Place images in: input/")
        print("\nSupported formats: JPG, PNG, WebP, HEIC")
        sys.exit(1)
    
    print(f"\n[SCAN] Found {len(image_files)} image(s)")
    
    # Process with progress
    total_start = time.time()
    total = len(image_files)
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        # Progress bar (ASCII compatible)
        percent = (idx / total) * 100
        bar_len = 25
        filled = int(bar_len * idx / total)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"\n[{bar}] {percent:.0f}% ({idx}/{total})")
        
        # Calculate output folder (maintain structure)
        try:
            rel_path = image_path.relative_to(input_dir)
            target_out = output_dir / rel_path.parent
            target_out.mkdir(parents=True, exist_ok=True)
            result = process_image(image_path, target_out)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((image_path, None, False))
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    total_time = time.time() - total_start
    success = sum(1 for r in results if r[2])
    failed = sum(1 for r in results if not r[2])
    
    print(f"\n  [+] Processed: {success}")
    print(f"  [-] Failed:    {failed}")
    print(f"  [TIME] Total:  {total_time:.1f}s")
    print(f"\n  [DIR] Output:  {output_dir}")
    print("\n" + "="*60)
    print("  Done! Images ready for Carousell upload.")
    print("="*60 + "\n")
    
    # Generate comparisons if requested
    if args.compare:
        comparison_dir = script_dir / "comparisons"
        generate_all_comparisons(results, comparison_dir)
    else:
        print("\n[INFO] Run with --compare to generate side-by-side comparisons")


if __name__ == "__main__":
    main()
