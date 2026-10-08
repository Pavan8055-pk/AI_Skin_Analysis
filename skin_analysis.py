import cv2
import numpy as np



def get_skin_mask(face_img):
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    mask[gray < 35] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask > 0


def analyze_skin(image, face):

    x, y, w, h = face

    mx, my = int(w * 0.12), int(h * 0.12)
    face_img = image[
        max(y + my, 0): y + h - my,
        max(x + mx, 0): x + w - mx
    ]

    if face_img.size == 0:
        return None

    face_img = cv2.resize(face_img, (200, 200))

    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)

    
    mask = get_skin_mask(face_img)
    if mask.sum() < 2000:
        mask = np.ones(gray.shape, dtype=bool)

    skin_pixels = mask.sum()

   
    brightness = np.mean(hsv[:, :, 2][mask])
    saturation = np.mean(hsv[:, :, 1][mask])

    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    texture = laplacian[mask].var()

   
    a_channel = lab[:, :, 1].astype(np.float32)
    a_skin = a_channel[mask]
    red_ratio = np.mean(a_skin > (np.median(a_skin) + 6)) * 100

    if red_ratio > 15:
        redness_status = "High"
    elif red_ratio > 6:
        redness_status = "Moderate"
    else:
        redness_status = "Low"

    
    l_channel = lab[:, :, 0].astype(np.float32)
    local_average = cv2.GaussianBlur(l_channel, (0, 0), 15)
    dark_spots = (local_average - l_channel) > 18
    dark_ratio = np.sum(dark_spots & mask) / skin_pixels * 100

    if dark_ratio > 8:
        pigmentation = "High"
    elif dark_ratio > 3:
        pigmentation = "Moderate"
    else:
        pigmentation = "Low"

    
    shine = (hsv[:, :, 2] > 230) & (hsv[:, :, 1] < 60)
    shine_ratio = np.sum(shine & mask) / skin_pixels * 100

    if shine_ratio > 6:
        skin_type = "Oily / Combination"
    elif shine_ratio > 1.5:
        skin_type = "Normal / Combination"
    else:
        skin_type = "Normal / Dry"

    
    score = 100
    score -= min(dark_ratio * 2, 25)              # dark spots
    score -= min(max(texture - 30, 0) * 0.1, 20)  # roughness
    score -= min(red_ratio * 1.0, 20)             # redness
    score -= min(shine_ratio * 1.0, 10)           # extra oil

    score = max(0, min(100, int(round(score))))

    
    return {
        "skin_type": skin_type,
        "brightness": round(float(brightness), 2),
        "saturation": round(float(saturation), 2),
        "texture": round(float(texture), 2),
        "redness": redness_status,
        "pigmentation": pigmentation,
        "skin_score": score
    }