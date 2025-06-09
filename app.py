import os
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# gdown 若未安裝則自動處理
try:
    import gdown
except ImportError:
    gdown = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# 模型路徑與網址
MODEL_URL = "https://drive.google.com/uc?export=download&id=1oIwiQ60jPQX0n75Tl_wCgsYyJU7_j_-M"
MODEL_PATH = "fitness_cnn_model.keras"
CLASS_NAMES = ['伏地挺身', '仰臥起坐', '側棒式', '深蹲']
IMG_SIZE = (128, 128)

# 自動下載模型（若不存在）
if not os.path.exists(MODEL_PATH):
    print("🔽 模型不存在，嘗試下載中...")
    try:
        if gdown:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        else:
            import requests
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        print("✅ 模型下載完成！")
    except Exception as e:
        print(f"❌ 模型下載失敗：{e}")
        raise RuntimeError("模型下載失敗，無法啟動伺服器")

# 載入模型
try:
    model = load_model(MODEL_PATH)
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗：{e}")
    raise RuntimeError("模型載入失敗")

@app.route('/')
def home():
    return '🏋️ 健身動作識別模型後端已啟動！'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 圖片處理
        if 'image' in request.files:
            file = request.files['image']
            if file.mimetype not in ['image/jpeg', 'image/png']:
                return jsonify({'error': '僅支援 JPG/PNG 格式圖片'}), 400
            image = Image.open(file).convert("RGB")

        elif request.is_json:
            data = request.get_json()
            base64_str = data.get('image', '')
            if not base64_str:
                return jsonify({'error': '請提供 base64 格式的 image 欄位'}), 400
            if ',' in base64_str:
                base64_str = base64_str.split(',')[-1]
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            return jsonify({'error': '請提供圖片（檔案或 base64）'}), 400

        # 圖像預處理
        image = image.resize(IMG_SIZE)
        arr = img_to_array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # 預測
        preds = model.predict(arr)[0]
        class_index = int(np.argmax(preds))
        class_name = CLASS_NAMES[class_index]
        confidence = float(preds[class_index])

        return jsonify({
            'class_name': class_name,
            'confidence': confidence
        })

    except Exception as e:
        # 印出 traceback 詳細錯誤除錯用
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
