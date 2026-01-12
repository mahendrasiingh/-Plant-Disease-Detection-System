from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

IMG_SIZE = 128

# Load class names from dataset folder
class_names = sorted(os.listdir("dataset"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["leaf_image"]

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)[0]
            raw_label = class_names[np.argmax(preds)]

            prediction = raw_label.replace("___", " ").replace("_", " ").title()
            confidence = round(float(np.max(preds)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
