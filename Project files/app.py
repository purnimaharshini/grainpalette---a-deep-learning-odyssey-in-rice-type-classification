from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/rice_model.h5')

# Get class names from train directory
class_names = sorted([d for d in os.listdir('dataset/train') if os.path.isdir(os.path.join('dataset/train', d))])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            img_path = 'uploads/' + img_file.filename
            os.makedirs('uploads', exist_ok=True)
            img_file.save(img_path)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            predicted_class = class_names[np.argmax(pred)]
            prediction = f"Predicted Rice Type: {predicted_class}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)