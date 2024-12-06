from flask import Flask, render_template, request, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__, static_folder='static')


# Modeli yukleyin
model = tf.keras.models.load_model("brain_tumor_model2.keras", compile=False)



# Gecici dosyalar için bir klasor belirleyin
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Resmi kaydet
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Resmi modelin giris boyutuna göre isleyin
            image = Image.open(filepath)

            # Resmin RGB formatında olup olmadığını kontrol edin ve gerekirse 
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_resized = image.resize((128, 128))
            image_array = np.array(image_resized, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)  # Batch boyutunu ekle

            # Tahmin yapın
            prediction = model.predict(image_array)
            confidence = prediction[0][0]
            result = "Tümör Var" if confidence > 0.5 else "Tümör Yok"

            #  URL'sini belirleyin
            image_url = url_for("static", filename=f"uploads/{file.filename}")

    return render_template("index.html", result=result, confidence=confidence, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)


