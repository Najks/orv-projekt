from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'C:\Users\nikda\project-rai-backend\uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['video']
    client_id = request.form['client_id']

    if not client_id:
        return jsonify({'error': 'No client_id provided'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        result = subprocess.check_output(['python', 'projekt_orv.py', file_path, client_id])
        return jsonify({'result': result.decode('utf-8')}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.output.decode('utf-8')}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)