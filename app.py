import os, base64, io, gc
import numpy as np
from PIL import Image, ImageFile
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 安全處理破損圖像（避免中斷）
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = None
model_path = "fitness_cnn_model.keras"
class_names = ['伏地挺身', '仰臥起坐', '側棒式', '深蹲']
IMG_SIZE = (96, 96)

def load_model_once():
    global model
    if model is None:
        print("🚀 載入模型中...")
        model = load_model(model_path, compile=False)
        print("✅ 模型已載入")

@app.route('/')
def home():
    return '🏋️ 後端運作中'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_model_once()

        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file).convert("RGB")
        elif request.is_json:
            data = request.get_json()
            base64_str = data.get('image', '').split(',')[-1]
            image = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")
        else:
            return jsonify({'error': '請提供圖片'}), 400

        image.thumbnail((400, 400))  # 限制圖片最大解析度（保護記憶體）
        image = image.resize(IMG_SIZE)
        arr = img_to_array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)[0]
        class_index = int(np.argmax(preds))
        class_name = class_names[class_index]
        confidence = float(preds[class_index])

        del image, arr, preds
        gc.collect()  # 強制回收

        return jsonify({
            'class_name': class_name,
            'confidence': confidence
        })

    except Exception as e:
        gc.collect()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
