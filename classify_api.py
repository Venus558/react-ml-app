from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from evaluation import evaluate
import glob

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'Uploaded_Images/user_upload/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(save_path)

    try:
        result = evaluate('Uploaded_Images/')
        for file in glob.glob(os.path.join(UPLOAD_FOLDER, "*")):
            os.remove(file)
        return jsonify({ "predictions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)