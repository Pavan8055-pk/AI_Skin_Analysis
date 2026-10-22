
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request
from skin_analysis import analyze_skin

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")

    if not file or not file.filename:
        return render_template(
            "index.html",
            error="Please select an image."
        )

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Upload a JPG, PNG or WEBP image."
        )

    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        return render_template(
            "index.html",
            error="Unable to read this image."
        )

    height, width = image.shape[:2]
    scale = 1000 / max(height, width)

    if scale < 1:
        image = cv2.resize(
            image, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(80, 80)
    )

    face_detected = len(faces) > 0
    skin_data = None
    preview = image.copy()

    if face_detected:
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = [int(v) for v in face]

        skin_data = analyze_skin(image, face)
        print("SKIN ANALYSIS DATA:", skin_data)

        cv2.rectangle(
            preview, (x, y), (x + w, y + h),
            (229, 70, 79), 3
        )

    ok, buffer = cv2.imencode(".jpg", preview)

    image_data = (
        base64.b64encode(buffer).decode("utf-8")
        if ok else None
    )

    return render_template(
        "index.html",
        face_detected=face_detected,
        image_data=image_data,
        skin_data=skin_data,
        error=None if face_detected else
        "No face detected. Try a clear, front-facing photo."
    )

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="Please upload an image under 8 MB."
    ), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
