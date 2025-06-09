import os
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 模型設定
model_path = "fitness_cnn_model.keras"
model = load_model(model_path)
class_names = ['伏地挺身', '仰臥起坐', '側棒式', '深蹲']
IMG_SIZE = (128, 128)

@app.route('/')
def home():
    return '🏋️ 健身動作識別模型後端已啟動！'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file).convert("RGB")
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': '請提供 base64 格式的 image 欄位'}), 400
            base64_str = data['image']
            if ',' in base64_str:
                base64_str = base64_str.split(',')[-1]
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            return jsonify({'error': '請提供圖片（檔案或 base64）'}), 400

        image = image.resize(IMG_SIZE)
        arr = img_to_array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        class_index = int(np.argmax(preds))
        class_name = class_names[class_index]
        confidence = float(preds[class_index])

        return jsonify({
            'class_name': class_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
