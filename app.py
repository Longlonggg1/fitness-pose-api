@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🔄 收到圖片，開始處理")  # 偵測開始

        # 你的既有程式碼...
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

        print("✅ 辨識完成")  # 偵測結束

        return jsonify({
            'class_name': class_name,
            'confidence': confidence
        })

    except Exception as e:
        print(f"❌ 預測錯誤: {e}")
        return jsonify({'error': str(e)}), 500
