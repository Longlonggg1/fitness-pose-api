import os
import base64
import io
import gdown
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# 模型參數
model_url = "https://drive.google.com/uc?export=download&id=1oIwiQ60jPQX0n75Tl_wCgsYyJU7_j_-M"
model_path = "fitness_cnn_model.keras"
gdrive_url = "https://drive.google.com/uc?id=1oIwiQ60jPQX0n75Tl_wCgsYyJU7_j_-M"

# 如果模型不存在就下載
if not os.path.exists(model_path):
    print("🔽 正在從 Google Drive 下載模型檔案...")
    try:
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("✅ 模型下載完成！")
    except Exception as e:
        print(f"❌ 模型下載失敗：{e}")
    gdown.download(gdrive_url, model_path, quiet=False)
    print("✅ 模型下載完成！")

# 載入模型
model = load_model(model_path)
class_names = ['伏地挺身', '仰臥起坐', '側棒式', '深蹲']
IMG_SIZE = (128, 128)

@app.route('/')
def home():
    return '🏋️ 健身動作識別模型後端已啟動！'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 方法一：multipart/form-data
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file).convert("RGB")

        # 方法二：application/json base64 字串
        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': '請提供 base64 格式的 image 欄位'}), 400
            
            base64_str = data['image']
            if ',' in base64_str:
                base64_str = base64_str.split(',')[-1]  # 移除開頭 "data:image/jpeg;base64,..."
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

        else:
            return jsonify({'error': '請提供圖片（檔案或 base64）'}), 400

        # 圖片預處理
        image = image.resize(IMG_SIZE)
        arr = img_to_array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # 預測
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

if _name_ == '__main__':
    app.run(debug=True)
