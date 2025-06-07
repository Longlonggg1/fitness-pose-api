import os
import gdown
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

model_path = "fitness_cnn_model.keras"
gdrive_url = "https://drive.google.com/uc?id=1oIwiQ60jPQX0n75Tl_wCgsYyJU7_j_-M"

if not os.path.exists(model_path):
    print("🔽 正在從 Google Drive 下載模型檔案...")
    gdown.download(gdrive_url, model_path, quiet=False)
    print("✅ 模型下載完成！")

model = load_model(model_path)

class_names = ['伏地挺身', '仰臥起坐', '側棒式', '深蹲']

IMG_SIZE = (128, 128)

@app.route('/')
def home():
    return '🏋️ 健身動作識別模型後端已啟動！'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '請上傳圖片'}), 400

    file = request.files['image']

    try:
        image = Image.open(file).convert("RGB")
        image = image.resize(IMG_SIZE)
        arr = img_to_array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        class_index = np.argmax(preds)
        class_name = class_names[class_index]
        confidence = float(preds[class_index])

        return jsonify({
            'class_name': class_name,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
