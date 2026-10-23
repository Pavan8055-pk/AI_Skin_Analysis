
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
    skin_score = quality_score

    recommendations = []
    if brightness > 145 and saturation > 70:
        recommendations.append(
            "Consider a gentle cleanser suitable for oily skin."
        )
    if redness > 20:
        recommendations.append(
            "Consider a fragrance-free moisturizer, Non-comedogenic sunscreen and gentle Facewash."
        )
    if not recommendations:
        recommendations.append(
            "Use a gentle cleanser, moisturizer and broad-spectrum sunscreen."
        )
        
    # Basic image-based pore feature estimate
    pore_gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    pore_gray = cv2.GaussianBlur(pore_gray, (5, 5), 0)

    local_mean = cv2.GaussianBlur(pore_gray, (15, 15), 0)
    dark_features = cv2.subtract(local_mean, pore_gray)

    _, pore_mask = cv2.threshold(
        dark_features, 8, 255, cv2.THRESH_BINARY
    )

    pore_mask = cv2.morphologyEx(
        pore_mask, cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    pore_feature_count, _ = cv2.connectedComponents(pore_mask)

    # This is a rough image-feature count, NOT a true pore count.
    pore_estimate = max(0, pore_feature_count - 1)

    
    # Experimental blackhead/whitehead candidate detection
    # These are visual feature estimates, not verified comedone counts.

    feature_gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    feature_gray = cv2.GaussianBlur(feature_gray, (3, 3), 0)

    feature_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )

    # Dark small features may resemble blackheads
    dark_response = cv2.morphologyEx(
        feature_gray, cv2.MORPH_BLACKHAT, feature_kernel
    )

    # Bright small features may resemble whiteheads
    bright_response = cv2.morphologyEx(
        feature_gray, cv2.MORPH_TOPHAT, feature_kernel
    )

    
    def count_candidate_features(response, threshold_value=30):
        # Filter weak image responses and reduce small noise
        _, mask = cv2.threshold(
            response, threshold_value, 255, cv2.THRESH_BINARY
        )

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        candidates = 0
        for i in range(1, count):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            if (
                4 <= area <= 30
                and 2 <= width <= 7
                and 2 <= height <= 7
            ):
                candidates += 1

        return candidates

    
    blackhead_estimate = count_candidate_features(dark_response)
    whitehead_estimate = count_candidate_features(bright_response)


    print("DEBUG: Reached final return")

    return {
        "skin_type": skin_type,
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "texture": round(texture, 2),
        "redness": round(redness, 2),
        "pigmentation": round(pigmentation_proxy, 2),
        "skin_score": skin_score,
        "confidence": 50,
        "image_quality": "Image-based estimate",
        "quality_score": quality_score,
        "quality_message": (
            "Results are approximate and depend on lighting and image quality."
        ),
        "redness_score": round(redness, 2),
        "shine_ratio": round(saturation, 2),
        "skin_coverage": "Estimated face region",
        "regions": {},
        "analysis_type": "Heuristic computer vision estimate",
        "medical_diagnosis": False,
        "blackhead_count": blackhead_estimate,
        "blackhead_status": "Approximate dark-feature count; not a true blackhead count",
        "whitehead_count": whitehead_estimate,
        "whitehead_status": "Approximate bright-feature count; not a true whitehead count",
        "pore_count": pore_estimate,
        "pore_status": "Approximate dark-feature count; not a true pore count",
        "pigmentation_map_status": "Image-based estimate",
        "pigmentation_map_coverage": round(pigmentation_proxy, 2),
        "pigmentation_map_regions": {},
        "dark_circle_status": "Not reliably assessed",
        "dark_circle_score": 38,
        "hydration_score": 60,
        "hydration_status": "Cannot be measured from an ordinary image",
        "estimated_skin_age": 20,
        "personalized_recommendations": recommendations,
    }