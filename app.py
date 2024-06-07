from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from projekt_orv import preprocess_dataset, augment_dataset, send_push_notification
from compare import compare_images
import requests

app = Flask(__name__)
# TODO nastavi dejanske mape
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['AUGMENTED_FOLDER'] = 'static/augmented'

# TODO isto tukaj da so iste s node.js streznikom
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUGMENTED_FOLDER'], exist_ok=True)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 201

@app.route('/process_video', methods=['POST'])
def process_video():
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File path is invalid or does not exist'}), 400

    # Klicanje funkcije za predobdelavo in augmentacijo
    preprocess_dataset(file_path, app.config['PROCESSED_FOLDER'])
    augment_dataset(app.config['PROCESSED_FOLDER'], app.config['AUGMENTED_FOLDER'])

    # Preverjanje identitete
    verification_result = compare_images(app.config['AUGMENTED_FOLDER'])
    
    return jsonify({'success': verification_result != 0, 'identity': verification_result}), 200

@app.route('/send_notification', methods=['POST'])
def send_notification():
    data = request.json
    registration_id = data.get('registration_id')
    title = data.get('title')
    message = data.get('message')

    # Pošiljanje obvestila preko Node.js API
    try:
        response = requests.post('http://localhost:3001/send-notification', json={
            'userId': registration_id,
            'title': title,
            'message': message
        })
        if response.status_code == 200:
            return jsonify({'message': 'Notification sent successfully'}), 200
        else:
            return jsonify({'error': 'Failed to send notification'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
