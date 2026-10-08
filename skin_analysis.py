import cv2
import numpy as np


# ============================================================
# SKINSENSE AI - IMPROVED COMPUTER VISION ANALYSIS
# ============================================================
#
# Important:
# This is an image-based computer vision estimator.
# It is NOT a medical diagnostic system.
#
# The algorithm analyzes:
# - skin appearance
# - brightness
# - saturation
# - visible redness
# - visible pigmentation-like regions
# - texture
# - shine
# - image quality
# - regional consistency
#
# ============================================================


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


# ------------------------------------------------------------
# 1. Improve image lighting
# ------------------------------------------------------------

def normalize_lighting(image):
    """
    Reduce the effect of uneven lighting using CLAHE
    on the luminance channel.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    normalized = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    return cv2.cvtColor(
        normalized,
        cv2.COLOR_LAB2BGR
    )


# ------------------------------------------------------------
# 2. Create better skin mask
# ------------------------------------------------------------

def get_skin_mask(face_img):

    ycrcb = cv2.cvtColor(
        face_img,
        cv2.COLOR_BGR2YCrCb
    )

    hsv = cv2.cvtColor(
        face_img,
        cv2.COLOR_BGR2HSV
    )

    # YCrCb skin range
    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([20, 133, 77]),
        np.array([245, 180, 135])
    )

    # HSV skin range
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 20, 35]),
        np.array([35, 220, 255])
    )

    # Combine masks
    mask = cv2.bitwise_and(
        ycrcb_mask,
        hsv_mask
    )

    # Remove very dark pixels
    gray = cv2.cvtColor(
        face_img,
        cv2.COLOR_BGR2GRAY
    )

    mask[gray < 30] = 0

    # Remove extremely bright pixels
    mask[gray > 250] = 0

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Slightly fill small gaps
    kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel2
    )

    return mask > 0


# ------------------------------------------------------------
# 3. Image quality analysis
# ------------------------------------------------------------

def analyze_image_quality(gray):

    brightness = float(np.mean(gray))

    contrast = float(np.std(gray))

    # Laplacian variance = approximate sharpness
    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    score = 100.0

    # Brightness
    if brightness < 45:
        score -= 30
    elif brightness < 65:
        score -= 15
    elif brightness > 235:
        score -= 20
    elif brightness > 220:
        score -= 10

    # Contrast
    if contrast < 20:
        score -= 20
    elif contrast < 30:
        score -= 10

    # Sharpness
    if sharpness < 30:
        score -= 30
    elif sharpness < 60:
        score -= 15

    score = clamp(score, 0, 100)

    if score >= 80:
        quality = "Good"
    elif score >= 60:
        quality = "Fair"
    else:
        quality = "Poor"

    return {
        "quality": quality,
        "quality_score": int(round(score)),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2)
    }


# ------------------------------------------------------------
# 4. Regional masks
# ------------------------------------------------------------

def create_region_masks(shape):

    h, w = shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]

    # Face center
    cx = w / 2
    cy = h / 2

    # Normalized coordinates
    nx = xx / w
    ny = yy / h

    # Forehead
    forehead = (
        (ny >= 0.05) &
        (ny < 0.32) &
        (nx > 0.22) &
        (nx < 0.78)
    )

    # Left cheek
    left_cheek = (
        (nx >= 0.08) &
        (nx < 0.42) &
        (ny >= 0.35) &
        (ny < 0.78)
    )

    # Right cheek
    right_cheek = (
        (nx > 0.58) &
        (nx <= 0.92) &
        (ny >= 0.35) &
        (ny < 0.78)
    )

    # Nose
    nose = (
        (nx >= 0.40) &
        (nx <= 0.60) &
        (ny >= 0.30) &
        (ny < 0.78)
    )

    # Chin
    chin = (
        (nx >= 0.30) &
        (nx <= 0.70) &
        (ny >= 0.72) &
        (ny <= 0.98)
    )

    return {
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "nose": nose,
        "chin": chin
    }


# ------------------------------------------------------------
# 5. Regional statistics
# ------------------------------------------------------------

def regional_statistics(lab, hsv, gray, skin_mask):

    regions = create_region_masks(
        gray.shape
    )

    results = {}

    for name, region in regions.items():

        valid = region & skin_mask

        pixel_count = int(valid.sum())

        if pixel_count < 100:
            results[name] = {
                "pixels": 0,
                "brightness": 0,
                "redness": 0,
                "texture": 0
            }
            continue

        brightness = float(
            np.mean(
                lab[:, :, 0][valid]
            )
        )

        redness = float(
            np.mean(
                lab[:, :, 1][valid]
            )
        )

        texture_region = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        texture = float(
            np.var(
                texture_region[valid]
            )
        )

        saturation = float(
            np.mean(
                hsv[:, :, 1][valid]
            )
        )

        results[name] = {
            "pixels": pixel_count,
            "brightness": round(brightness, 2),
            "redness": round(redness, 2),
            "texture": round(texture, 2),
            "saturation": round(saturation, 2)
        }

    return results


# ------------------------------------------------------------
# 6. Redness estimation
# ------------------------------------------------------------

def calculate_redness(lab, skin_mask):

    a_channel = lab[:, :, 1].astype(
        np.float32
    )

    values = a_channel[skin_mask]

    if values.size == 0:
        return 0.0, "Unknown"

    median = np.median(values)

    # Adaptive threshold instead of fixed threshold
    threshold = median + 5

    red_pixels = (
        (a_channel > threshold) &
        skin_mask
    )

    ratio = (
        np.sum(red_pixels) /
        max(np.sum(skin_mask), 1)
    ) * 100

    # Normalize ratio
    redness_score = clamp(
        ratio * 4.0,
        0,
        100
    )

    if redness_score >= 65:
        status = "High"
    elif redness_score >= 35:
        status = "Moderate"
    else:
        status = "Low"

    return float(redness_score), status


# ------------------------------------------------------------
# 7. Pigmentation estimation
# ------------------------------------------------------------

def calculate_pigmentation(
    lab,
    gray,
    skin_mask
):

    l_channel = lab[:, :, 0].astype(
        np.float32
    )

    # Local illumination baseline
    local_average = cv2.GaussianBlur(
        l_channel,
        (0, 0),
        15
    )

    # Relative darkness
    darkness = (
        local_average -
        l_channel
    )

    dark_regions = (
        (darkness > 12) &
        skin_mask
    )

    skin_pixels = max(
        int(np.sum(skin_mask)),
        1
    )

    ratio = (
        np.sum(dark_regions) /
        skin_pixels
    ) * 100

    score = clamp(
        ratio * 5.0,
        0,
        100
    )

    if score >= 65:
        status = "High"
    elif score >= 30:
        status = "Moderate"
    else:
        status = "Low"

    return float(score), status


# ------------------------------------------------------------
# 8. Shine / skin type estimation
# ------------------------------------------------------------

def estimate_skin_type(
    hsv,
    skin_mask
):

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Strong shine pixels
    shine = (
        (value > 215) &
        (saturation < 75) &
        skin_mask
    )

    skin_pixels = max(
        int(np.sum(skin_mask)),
        1
    )

    shine_ratio = (
        np.sum(shine) /
        skin_pixels
    ) * 100

    # More stable classification
    if shine_ratio >= 7:
        skin_type = "Oily"
    elif shine_ratio >= 3:
        skin_type = "Combination"
    elif shine_ratio >= 1:
        skin_type = "Normal"
    else:
        skin_type = "Normal / Dry"

    return (
        skin_type,
        float(shine_ratio)
    )


# ------------------------------------------------------------
# 9. Texture analysis
# ------------------------------------------------------------

def calculate_texture(
    gray,
    skin_mask
):

    # Remove high-frequency sensor noise
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    laplacian = cv2.Laplacian(
        blurred,
        cv2.CV_64F
    )

    values = np.abs(
        laplacian[skin_mask]
    )

    if values.size == 0:
        return 0.0

    # Percentile is more stable than raw variance
    texture = float(
        np.percentile(values, 75)
    )

    return texture


# ------------------------------------------------------------
# 10. Overall skin score
# ------------------------------------------------------------

def calculate_skin_score(
    redness_score,
    pigmentation_score,
    texture,
    shine_ratio,
    quality_score
):

    score = 100.0

    # Redness contribution
    score -= redness_score * 0.20

    # Pigmentation contribution
    score -= pigmentation_score * 0.22

    # Texture contribution
    texture_penalty = clamp(
        (texture - 8) * 1.2,
        0,
        20
    )

    score -= texture_penalty

    # Excessive shine
    shine_penalty = clamp(
        (shine_ratio - 2) * 0.7,
        0,
        8
    )

    score -= shine_penalty

    # Poor image quality slightly reduces confidence,
    # not the underlying skin condition.
    quality_penalty = max(
        0,
        (70 - quality_score) * 0.08
    )

    score -= quality_penalty

    score = clamp(
        score,
        0,
        100
    )

    return int(round(score))


# ------------------------------------------------------------
# 11. Confidence estimation
# ------------------------------------------------------------

def calculate_confidence(
    quality_score,
    skin_coverage,
    regional_count
):

    confidence = 0.0

    # Image quality
    confidence += quality_score * 0.55

    # Skin coverage
    coverage_score = clamp(
        skin_coverage * 100,
        0,
        100
    )

    confidence += coverage_score * 0.30

    # Regional analysis availability
    region_score = clamp(
        regional_count / 5 * 100,
        0,
        100
    )

    confidence += region_score * 0.15

    return int(
        round(
            clamp(
                confidence,
                0,
                100
            )
        )
    )


# ------------------------------------------------------------
# 12. Main analysis function
# ------------------------------------------------------------

def analyze_skin(image, face):

    if image is None or face is None:
        return None

    x, y, w, h = [
        int(value)
        for value in face
    ]

    # -----------------------------------------
    # Expand slightly around face
    # -----------------------------------------

    mx = int(w * 0.08)
    my = int(h * 0.08)

    x1 = max(x + mx, 0)
    y1 = max(y + my, 0)

    x2 = min(
        x + w - mx,
        image.shape[1]
    )

    y2 = min(
        y + h - my,
        image.shape[0]
    )

    face_img = image[
        y1:y2,
        x1:x2
    ]

    if face_img.size == 0:
        return None

    # -----------------------------------------
    # Resize consistently
    # -----------------------------------------

    face_img = cv2.resize(
        face_img,
        (300, 300),
        interpolation=cv2.INTER_AREA
    )

    # -----------------------------------------
    # Normalize lighting
    # -----------------------------------------

    normalized = normalize_lighting(
        face_img
    )

    # -----------------------------------------
    # Color spaces
    # -----------------------------------------

    hsv = cv2.cvtColor(
        normalized,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        normalized,
        cv2.COLOR_BGR2GRAY
    )

    lab = cv2.cvtColor(
        normalized,
        cv2.COLOR_BGR2LAB
    )

    # -----------------------------------------
    # Image quality
    # -----------------------------------------

    quality = analyze_image_quality(
        gray
    )

    # -----------------------------------------
    # Skin segmentation
    # -----------------------------------------

    skin_mask = get_skin_mask(
        normalized
    )

    skin_pixels = int(
        np.sum(skin_mask)
    )

    total_pixels = skin_mask.size

    skin_coverage = (
        skin_pixels /
        max(total_pixels, 1)
    )

    # If segmentation fails, don't produce
    # misleading results.
    if skin_coverage < 0.08:

        return {
            "skin_type": "Unable to determine",
            "brightness": 0,
            "saturation": 0,
            "texture": 0,
            "redness": "Unknown",
            "pigmentation": "Unknown",
            "skin_score": 0,
            "confidence": 20,
            "image_quality": quality["quality"],
            "quality_score": quality["quality_score"],
            "message": (
                "Skin region could not be detected "
                "reliably. Try a clearer photo."
            )
        }

    # -----------------------------------------
    # Global measurements
    # -----------------------------------------

    brightness = float(
        np.mean(
            hsv[:, :, 2][skin_mask]
        )
    )

    saturation = float(
        np.mean(
            hsv[:, :, 1][skin_mask]
        )
    )

    # -----------------------------------------
    # Skin type
    # -----------------------------------------

    skin_type, shine_ratio = (
        estimate_skin_type(
            hsv,
            skin_mask
        )
    )

    # -----------------------------------------
    # Texture
    # -----------------------------------------

    texture = calculate_texture(
        gray,
        skin_mask
    )

    # -----------------------------------------
    # Redness
    # -----------------------------------------

    redness_score, redness_status = (
        calculate_redness(
            lab,
            skin_mask
        )
    )

    # -----------------------------------------
    # Pigmentation
    # -----------------------------------------

    pigmentation_score, pigmentation_status = (
        calculate_pigmentation(
            lab,
            gray,
            skin_mask
        )
    )

    # -----------------------------------------
    # Regional analysis
    # -----------------------------------------

    regional = regional_statistics(
        lab,
        hsv,
        gray,
        skin_mask
    )

    available_regions = sum(
        1
        for region in regional.values()
        if region["pixels"] >= 100
    )

    # -----------------------------------------
    # Overall score
    # -----------------------------------------

    skin_score = calculate_skin_score(
        redness_score,
        pigmentation_score,
        texture,
        shine_ratio,
        quality["quality_score"]
    )

    # -----------------------------------------
    # Confidence
    # -----------------------------------------

    confidence = calculate_confidence(
        quality["quality_score"],
        skin_coverage,
        available_regions
    )

    # -----------------------------------------
    # Quality message
    # -----------------------------------------

    if quality["quality"] == "Poor":

        quality_message = (
            "Photo quality is low. "
            "Use bright, even lighting and "
            "keep your face clearly visible."
        )

    elif quality["quality"] == "Fair":

        quality_message = (
            "Photo quality is acceptable. "
            "Better lighting may improve reliability."
        )

    else:

        quality_message = (
            "Photo quality is good for image-based analysis."
        )

    # -----------------------------------------
    # Return result
    # -----------------------------------------

    return {

        # Main results
        "skin_type": skin_type,

        "brightness": round(
            brightness,
            2
        ),

        "saturation": round(
            saturation,
            2
        ),

        "texture": round(
            texture,
            2
        ),

        "redness": redness_status,

        "pigmentation": pigmentation_status,

        "skin_score": skin_score,

        # New information
        "confidence": confidence,

        "image_quality": quality["quality"],

        "quality_score": quality[
            "quality_score"
        ],

        "quality_message": quality_message,

        # Supporting measurements
        "redness_score": round(
            redness_score,
            2
        ),

        "pigmentation_score": round(
            pigmentation_score,
            2
        ),

        "shine_ratio": round(
            shine_ratio,
            2
        ),

        "skin_coverage": round(
            skin_coverage * 100,
            2
        ),

        # Regional analysis
        "regions": regional,

        # Safety/interpretation
        "analysis_type": (
            "Image-based computer vision estimate"
        ),

        "medical_diagnosis": False
    }