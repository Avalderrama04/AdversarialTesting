from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import time
import torchvision
from model_handler import MRI_Classifier
import logging
import traceback
import io
from PIL import Image
import json
import sys
import google.generativeai as genai  
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = MRI_Classifier('best_densenet_model.pth')

PROJECT_ID = "mriclassifier"
REGION = "us-central1"
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
gemini_chat = gemini_model.start_chat(history=[])



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_gemini_diagnostic(image_bytes, classification_label, is_adversarial, anomaly_score):
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = [
            image,
            f"This is an MRI brain tumor image. The classification result is: {classification_label}.",
            f"Adversarial status: {'Potentially adversarial' if is_adversarial else 'Likely not adversarial'}.",
            f"Anomaly score (if adversarial): {anomaly_score:.4f}.",
            "Provide a concise diagnostic explanation in JSON format with ONLY this exact structure:",
            "{",
            '"key_findings": "...",',
            '"potential_implications": "...",',
            '"recommended_next_steps": "..."',
            "}",
            "Return ONLY the JSON object with no additional text, explanations, or markdown formatting."
        ]

        response = gemini_chat.send_message(prompt)
        
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            start_idx = response.text.find('{')
            end_idx = response.text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response.text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {
                    "error": "Could not extract JSON from response",
                    "raw_response": response.text
                }
    except Exception as e:
        logger.error(f"Error in Gemini diagnostic: {str(e)}")
        return {
            "error": "Error generating diagnostic",
            "exception": str(e)
        }

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    user_message = request.json.get('message')
    try:
        response = gemini_chat.send_message(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        logger.error("Error in Gemini chat:")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Unable to get response from Gemini'}), 500

@app.route('/predict', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        timestamp = str(int(time.time()))
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = model.predict(filepath)
        print(f"[DEBUG] Flask response anomaly status: {result['anomaly']}", file=sys.stdout, flush=True)

        display_path = os.path.join(app.config['UPLOAD_FOLDER'], f"display_{filename}")
        torchvision.utils.save_image(result['display_image'], display_path)

        probabilities = {model.class_names[i]: f"{prob*100:.2f}%"
                        for i, prob in enumerate(result['probabilities'])}

        with open(filepath, "rb") as image_file:
            image_bytes = image_file.read()

        diagnostic_info = get_gemini_diagnostic(
            image_bytes,
            result['class'],
            result['anomaly']['is_anomalous'],
            result['anomaly']['anomaly_score']
        )

        return jsonify({
            'status': 'success',
            'prediction': result['class'],
            'confidence': f"{result['confidence']*100:.2f}%",
            'probabilities': probabilities,
            'display_image': f"/display/{filename}",
            'is_adversarial': result['anomaly']['is_anomalous'],
            'anomaly_score': float(result['anomaly']['anomaly_score']),
            'diagnostic_info': diagnostic_info  
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/display/<filename>')
def serve_display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], f"display_{filename}")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5004, debug=True)