# app.py

from flask import Flask, request, jsonify
import requests
import os
import cv2
import time
import traceback
from compress import read_grayscale_bmp, Compress, write_compressed_file
from decompress import read_compressed_file, Decompress, write_decompressed_bmp
from compare import compare_images
from projekt_orv import preprocess_image, augment_image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'project-rai-backend', 'uploads'))
app.config['FRAME_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frames'))
app.config['PROCESSED_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed'))
app.config['AUGMENTED_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'augmented'))
app.config['COMPRESSED_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'compressed'))
app.config['TEMP_DECOMPRESSED_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp_decompressed'))

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAME_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUGMENTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['COMPRESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_DECOMPRESSED_FOLDER'], exist_ok=True)

@app.route('/')
def hello_world():
    return 'Hello, World!'

def compress_image(input_image, compressed_dir):
    """
    Compress the given image and save the compressed file to the COMPRESSED_FOLDER.
    Returns the path to the compressed file.
    """
    try:
        P, X, Y = read_grayscale_bmp(input_image)
        original_size = os.path.getsize(input_image)
        output_compressed = os.path.join(compressed_dir, f"compressed_{os.path.splitext(os.path.basename(input_image))[0]}.bin")

        start_time = time.time()
        compressed_data = Compress(P, X, Y)
        compression_time = time.time() - start_time

        write_compressed_file(compressed_data, output_compressed)
        compressed_size = os.path.getsize(output_compressed)
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

        print(f"Compressed {input_image} to {output_compressed} in {compression_time:.4f} seconds")
        print(f"Original size: {original_size} bytes, Compressed size: {compressed_size} bytes, Compression Ratio: {ratio:.2f}")

        return output_compressed
    except Exception as e:
        print(f"[ERROR] Error compressing image {input_image}: {e}")
        traceback.print_exc()
        return None

def decompress_image(compressed_file, decompressed_dir):
    """
    Decompress the given compressed file and save the decompressed image to the TEMP_DECOMPRESSED_FOLDER.
    Returns the path to the decompressed file.
    """
    try:
        compressed_data = read_compressed_file(compressed_file)
        start_time = time.time()
        P = Decompress(compressed_data)
        decompression_time = time.time() - start_time

        output_decompressed = os.path.join(
            decompressed_dir, f"decompressed_{os.path.splitext(os.path.basename(compressed_file))[0]}.bmp")
        write_decompressed_bmp(P, output_decompressed)

        print(f"Decompressed {compressed_file} to {output_decompressed} in {decompression_time:.4f} seconds")
        return output_decompressed
    except Exception as e:
        print(f"[ERROR] Error decompressing file {compressed_file}: {e}")
        traceback.print_exc()
        return None

def extract_and_compress_frames(video_path, num_frames=1):
    """
    Extract frames from the video, compress them, and save to the COMPRESSED_FOLDER.
    Deletes the original extracted .jpg frames after compression.
    Returns a list of paths to the compressed files.
    """
    print(f"Extracting frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    frame_interval = max(total_frames // num_frames, 1)
    compressed_files = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(app.config['FRAME_FOLDER'], f"frame_{i}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Extracted frame {i}: {frame_filename}")

            # Compress the extracted frame
            compressed_file = compress_image(frame_filename, app.config['COMPRESSED_FOLDER'])
            if compressed_file:
                compressed_files.append(compressed_file)
                # Remove the original .jpg frame after compression
                try:
                    os.remove(frame_filename)
                    print(f"Removed original frame file: {frame_filename}")
                except Exception as e:
                    print(f"[ERROR] Error removing file {frame_filename}: {e}")
        else:
            print(f"[WARNING] Failed to extract frame {i}")

    cap.release()
    print(f"All compressed frames: {compressed_files}")
    return compressed_files

def decompress_and_prepare_frames(compressed_files):
    """
    Decompress the given list of compressed files and save them to the TEMP_DECOMPRESSED_FOLDER.
    Returns a list of paths to the decompressed .bmp files.
    """
    print(f"Decompressing {len(compressed_files)} compressed files for processing.")
    decompressed_files = []

    for compressed_file in compressed_files:
        decompressed_file = decompress_image(compressed_file, app.config['TEMP_DECOMPRESSED_FOLDER'])
        if decompressed_file:
            decompressed_files.append(decompressed_file)

    print(f"Decompressed files: {decompressed_files}")
    return decompressed_files

def process_decompressed_file(decompressed_file, idx):
    """
    Preprocess and augment the decompressed frame.
    Saves the processed and augmented images.
    Returns the path to the processed file and a list of augmented file paths.
    """
    try:
        # Preprocess the decompressed frame
        processed_image = preprocess_image(decompressed_file)
        processed_filename = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_frame_{idx}.jpg")
        cv2.imwrite(processed_filename, processed_image)
        print(f"Processed frame {idx}: {processed_filename}")

        # Augment the processed frame
        augmented_images = augment_image(processed_image)
        augmented_filenames = []
        for j, aug_img in enumerate(augmented_images):
            augmented_filename = os.path.join(app.config['AUGMENTED_FOLDER'], f"augmented_frame_{idx}_{j}.jpg")
            cv2.imwrite(augmented_filename, aug_img)
            print(f"Augmented frame {idx}_{j}: {augmented_filename}")
            augmented_filenames.append(augmented_filename)

        return processed_filename, augmented_filenames
    except Exception as e:
        print(f"[ERROR] Error processing decompressed file {decompressed_file}: {e}")
        traceback.print_exc()
        return None, []

def cleanup_files(file_list, keep_compressed=False):
    """
    Remove specified files from the filesystem.
    If keep_compressed is True, files in the COMPRESSED_FOLDER are not removed.
    """
    for file_path in file_list:
        try:
            if keep_compressed and file_path.startswith(app.config['COMPRESSED_FOLDER']):
                print(f"Retaining compressed file: {file_path}")
                continue
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"[ERROR] Error removing file {file_path}: {e}")

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Handle video upload, extract and compress frames, decompress for processing, preprocess, augment,
    compare images, and cleanup temporary files.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Secure the filename to prevent directory traversal attacks
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(f"Received request to process video. File path: {file_path}")

    print(f"Processing video file at path: {file_path}")

    # Step 1: Extract and compress frames
    num_frames = 1  # Adjust as needed for the number of frames to extract
    compressed_files = extract_and_compress_frames(file_path, num_frames=num_frames)
    if not compressed_files:
        print("[ERROR] Failed to extract and compress frames from video")
        return jsonify({'error': 'Failed to extract and compress frames from video'}), 500

    # Step 2: Decompress frames for processing
    decompressed_files = decompress_and_prepare_frames(compressed_files)
    if not decompressed_files:
        print("[ERROR] Failed to decompress frames for processing")
        return jsonify({'error': 'Failed to decompress frames for processing'}), 500

    # Step 3: Preprocess and augment decompressed frames
    processed_files = []
    augmented_files = []
    for idx, decompressed_file in enumerate(decompressed_files):
        processed_filename, augmented_filenames = process_decompressed_file(decompressed_file, idx)
        if processed_filename:
            processed_files.append(processed_filename)
            augmented_files.extend(augmented_filenames)

    if not augmented_files:
        print("[ERROR] No augmented files available for comparison")
        return jsonify({'error': 'No augmented files available for comparison'}), 500

    # Step 4: Compare augmented images
    verification_result = compare_images(augmented_files)
    print(f"Verification result from compare_images: {verification_result}")

    # Step 5: Cleanup temporary and processed files
    files_to_cleanup = decompressed_files + processed_files + augmented_files
    cleanup_files(files_to_cleanup, keep_compressed=True)  # Retain compressed files

    # Optionally, remove the uploaded video file after processing
    try:
        os.remove(file_path)
        print(f"Removed uploaded video file: {file_path}")
    except Exception as e:
        print(f"[ERROR] Error removing uploaded video file {file_path}: {e}")

    # Respond with the verification result
    if verification_result == 0:
        print("No match found")
        return jsonify({'success': True, 'identity': 0}), 200
    else:
        print(f"Match found: Identity {verification_result}")
        return jsonify({'success': True, 'identity': verification_result}), 200

@app.route('/send_notification', methods=['POST'])
def send_notification():
    """
    Send a notification to a user via the backend service.
    """
    data = request.json
    registration_id = data.get('registration_id')
    title = data.get('title')
    message = data.get('message')
    print(f"Sending notification to {registration_id} with title '{title}' and message '{message}'")

    try:
        response = requests.post('http://backend:3001/send-notification', json={
            'userId': registration_id,
            'title': title,
            'message': message
        })
        if response.status_code == 200:
            print("Notification sent successfully")
            return jsonify({'message': 'Notification sent successfully'}), 200
        else:
            print(f"Failed to send notification. Status code: {response.status_code}")
            return jsonify({'error': 'Failed to send notification'}), response.status_code
    except Exception as e:
        print(f"[ERROR] Error sending notification: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/list_uploads', methods=['GET'])
def list_uploads():
    """
    List all files in the UPLOAD_FOLDER.
    """
    print("Received request to list files in 'uploads' folder")
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        print(f"Files in 'uploads' folder: {files}")
        return jsonify({'files': files}), 200
    except Exception as e:
        print(f"[ERROR] Error listing files in 'uploads' folder: {e}")
        return jsonify({'error': str(e)}), 500

def print_uploads_on_startup():
    """
    Print the list of files in the UPLOAD_FOLDER when the server starts.
    """
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        print("Files in 'uploads' folder at startup:")
        for file in files:
            print(file)
    except Exception as e:
        print(f"[ERROR] Error listing files in 'uploads' folder: {e}")

if __name__ == '__main__':
    print_uploads_on_startup()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
