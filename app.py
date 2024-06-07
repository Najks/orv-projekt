from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

# Nastavite mapo za nalaganje videoposnetkov
UPLOAD_FOLDER = os.path.join('C:', 'Users', 'nikda', 'project-rai-backend', 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Endpoint za preverjanje identitete
@app.route('/verify_video', methods=['POST'])
def verify_video():
    # Pridobite ime datoteke in client_id iz zahteve
    video_filename = request.form.get('video_filename')
    client_id = request.form.get('client_id')

    # Preverite, če so vse potrebne informacije prisotne
    if not video_filename or not client_id:
        return jsonify({'error': 'Missing required fields'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

    # Preverite, če videoposnetek obstaja
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404

    try:
        # Zamenjajta 'projekt_orv.py z dejansko dat ki trenira model'
        result = subprocess.check_output(['python', 'projekt_orv.py', video_path, client_id])
        return jsonify({'result': result.decode('utf-8')}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.output.decode('utf-8')}), 500

if __name__ == '__main__':
    # Zagon Flask aplikacije
    app.run(debug=True, port=5000)
