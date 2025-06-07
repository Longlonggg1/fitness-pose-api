from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# 載入你的模型
model = load_model("fitness_cnn_model.keras")

# 替換為你資料夾中的類別名稱（根據 flow_from_directory 自動產生的順序）
class_names = ['伏地挺身', '仰臥起坐', '側棒式', '深蹲']  # ← 根據你的資料夾名稱排序可能不同，請照 val_generator.class_indices 排序

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
