
import cv2
import numpy as np


def analyze_skin(image, face):
    x, y, w, h = [int(v) for v in face]

    height, width = image.shape[:2]

    # Crop the detected face safely
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    face_image = image[y1:y2, x1:x2]

    if face_image.size == 0:
        return {
            "skin_type": "Unable to determine",
            "skin_score": 0,
            "confidence": 0,
            "image_quality": "Poor"
        }

    hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    saturation = float(np.mean(hsv[:, :, 1]))

    if saturation > 90 and brightness > 140:
        skin_type = "Oily (possible)"
    elif brightness < 100:
        skin_type = "Low brightness"
    else:
        skin_type = "Normal / combination (possible)"

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    texture = float(np.std(gray))

    skin_score = max(
        0, min(100, int(100 - abs(texture - 35)))
    )

    return {
        "skin_type": skin_type,
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "texture": round(texture, 2),
        "redness": 0,
        "pigmentation": 0,
        "skin_score": skin_score,
        "confidence": 50,
        "image_quality": "Image-based estimate",
        "quality_score": 50,
        "quality_message": (
            "Results are approximate and depend on lighting."
        ),
        "analysis_type": "Heuristic computer vision estimate",
        "medical_diagnosis": False
    }
