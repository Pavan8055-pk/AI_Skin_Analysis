import base64

import cv2
import numpy as np
from flask import Flask, render_template, request

from skin_analysis import analyze_skin


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_SIDE = 1000  # large photos are shrunk to this size (faster + consistent)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def resize_if_large(image):
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return image


def detect_face(image):
    """Return the biggest face (x, y, w, h) or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # helps in dim / uneven light

    min_size = max(80, int(min(image.shape[:2]) * 0.15))

    for neighbors in (5, 3):
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=neighbors,
            minSize=(min_size, min_size)
        )
        if len(faces) > 0:
            # Choose the largest face, not just the first one
            return max(faces, key=lambda f: f[2] * f[3])

    return None


def render_error(message, code=400):
    return render_template("index.html", error=message), code



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files.get("image")

    if file is None or file.filename == "":
        return render_error("Please select an image.")

    if not allowed_file(file.filename):
        return render_error("Please upload a JPG, PNG or WEBP image.")

    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        return render_error("Unable to read this image. Try another photo.")

    image = resize_if_large(image)

    face = detect_face(image)
    face_detected = face is not None

    skin_data = None
    if face_detected:
        skin_data = analyze_skin(image, face)

    preview = image.copy()
    if face_detected:
        x, y, w, h = [int(v) for v in face]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (229, 70, 79), 3)

    ok, buffer = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_data = base64.b64encode(buffer).decode("utf-8") if ok else None

    return render_template(
        "index.html",
        face_detected=face_detected,
        image_data=image_data,
        skin_data=skin_data
    )


@app.errorhandler(413)
def file_too_large(e):
    return render_error("Image is too large. Please upload a photo under 8 MB.", 413)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )