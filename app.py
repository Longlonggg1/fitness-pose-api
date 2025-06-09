import os
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 新增 gdown
try:
    import gdown
except ImportError:
    gdown = None

app = Flask(__name__)
CORS(app)  # 啟用跨域

# 模型設定
model_url = "https://drive.google.com/uc?export=download&id=1oIwiQ60jPQX0n75Tl_wCgsYyJU7_j_-M"
model_path = "fitness_cnn_model.keras"

# 如果模型不存在，嘗試下載
if not os.path.exists(model_path):
    print("🔽 正在下載模型檔案...")
    try:
        if gdown:
            # 用 gdown 下載
            gdown.download(model_url, model_path, quiet=False)
        else:
            # 如果沒有 gdown，用 requests 備用下載
            import requests
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
        print("✅ 模型下載完成！")
    except Exception as e:
        print(f"❌ 模型下載失敗：{e}")

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
        # multipart/form-data 傳檔案
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file).convert("RGB")

        # application/json 傳 base64 字串
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
