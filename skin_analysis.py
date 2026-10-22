
import cv2
import numpy as np


def analyze_skin(image, face):
    x, y, w, h = map(int, face)

    height, width = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)

    face_img = image[y1:y2, x1:x2]

    if face_img.size == 0:
        raise ValueError("Unable to analyse the detected face.")

    # Approximate skin region, excluding some outer face areas.
    fh, fw = face_img.shape[:2]
    skin = face_img[
        int(fh * 0.20):int(fh * 0.82),
        int(fw * 0.15):int(fw * 0.85)
    ]

    if skin.size == 0:
        skin = face_img

    hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(hsv[:, :, 2]))
    saturation = float(np.mean(hsv[:, :, 1]))
    texture = float(np.std(gray))

    # These are image-based proxies, not medical measurements.
    redness = float(np.mean(skin[:, :, 2]) - np.mean(skin[:, :, 0]))
    redness = round(max(0, min(100, redness)), 2)

    skin_type = (
        "Oily (possible)" if saturation > 75 and brightness > 130
        else "Dry (possible)" if brightness < 100
        else "Normal / combination (possible)"
    )

    # Detect dark local spots and bright local spots heuristically.
    smooth = cv2.GaussianBlur(gray, (9, 9), 0)
    dark_difference = cv2.subtract(smooth, gray)
    bright_difference = cv2.subtract(gray, smooth)

    _, dark_mask = cv2.threshold(
        dark_difference, 18, 255, cv2.THRESH_BINARY
    )
    _, bright_mask = cv2.threshold(
        bright_difference, 18, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(
        dark_mask, cv2.MORPH_OPEN, kernel
    )
    bright_mask = cv2.morphologyEx(
        bright_mask, cv2.MORPH_OPEN, kernel
    )

    dark_count = int(cv2.countNonZero(dark_mask))
    bright_count = int(cv2.countNonZero(bright_mask))
    total_pixels = max(1, gray.size)

    blackhead_ratio = dark_count / total_pixels * 100
    whitehead_ratio = bright_count / total_pixels * 100

    # Approximate pore visibility from local texture.
    pore_proxy = float(np.std(gray))

    # Approximate pigmentation variation.
    pigmentation_proxy = float(np.std(gray))

    quality_score = int(max(0, min(
        100, 100 - abs(texture - 25) * 1.5
    )))

    recommendations = []
    if brightness > 145 and saturation > 70:
        recommendations.append(
            "Consider a gentle cleanser suitable for oily skin."
        )
    if redness > 20:
        recommendations.append(
            "Consider a fragrance-free moisturizer and gentle skincare."
        )
    if not recommendations:
        recommendations.append(
            "Use a gentle cleanser, moisturizer and broad-spectrum sunscreen."
        )

    return {
        "skin_type": skin_type,
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "texture": round(texture, 2),
        "redness": redness,
        "pigmentation": round(pigmentation_proxy, 2),
        "skin_score": quality_score,
        "confidence": 50,
        "image_quality": "Image-based estimate",
        "quality_score": quality_score,
        "quality_message": "Results depend on lighting and image quality.",
        "redness_score": redness,
        "shine_ratio": round(saturation, 2),
        "skin_coverage": "Estimated face region",
        "regions": {},
        "analysis_type": "Heuristic computer vision estimate",
        "medical_diagnosis": False,

        "blackhead_count": dark_count,
        "blackhead_status": f"Dark-spot candidates: {blackhead_ratio:.2f}%",
        "whitehead_count": bright_count,
        "whitehead_status": f"Bright-spot candidates: {whitehead_ratio:.2f}%",
        "pore_count": round(pore_proxy, 2),
        "pore_status": "Texture-based visibility proxy",
        "pigmentation_map_status": "Image-based estimate",
        "pigmentation_map_coverage": round(pigmentation_proxy, 2),
        "pigmentation_map_regions": {},
        "dark_circle_status": "Not reliably assessed",
        "dark_circle_score": None,
        "hydration_score": None,
        "hydration_status": "Cannot be measured from an ordinary photo",
        "estimated_skin_age": None,
        "personalized_recommendations": recommendations
    }
