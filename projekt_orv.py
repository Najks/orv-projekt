import cv2
import os
import numpy as np
from pyfcm import FCMNotification
from pymongo import MongoClient
import gridfs


def capture_video_and_extract_frames(user_id, duration=3, save_path='dataset'):
    # Ustvari mapo, če ne obstaja
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Inicializacija kamere
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frekvenčna hitrost (število slik na sekundo)
    total_frames = int(duration * fps)  # Skupno število slik

    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        # Shrani vsako frejmo kot sliko
        img_name = f"{save_path}/user_{user_id}_{frame_count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved.")
        
        frame_count += 1

        # Pokaži okno z zajeto sliko
        cv2.imshow("Capture", frame)

        # Prekini zajemanje s tipko 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def preprocess_dataset(dataset_path='dataset', processed_path='processed'):
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        processed_img = preprocess_image(img_path)
        
        # Shrani obdelano sliko
        processed_img_path = os.path.join(processed_path, filename)
        cv2.imwrite(processed_img_path, processed_img)
        print(f"{processed_img_path} saved.")




def augment_image(image):
    augmented_images = []

    # Horizontal flip
    flip_horizontal = np.fliplr(image)
    augmented_images.append(flip_horizontal)
    
    # Increase brightness
    bright_image = np.clip(image * 1.2 + 30, 0, 255).astype(np.uint8)
    augmented_images.append(bright_image)
    
    # Increase contrast
    contrast_image = np.clip(image * 1.5, 0, 255).astype(np.uint8)
    augmented_images.append(contrast_image)
    
    # Rotate image by a random angle between -10 and 10 degrees
    rows, cols = image.shape[:2]
    angle = np.random.uniform(-10, 10)
    M = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                  [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
                  [0, 0, 1]])
    M[0, 2] = (cols - cols * np.cos(np.deg2rad(angle)) + rows * np.sin(np.deg2rad(angle))) / 2
    M[1, 2] = (rows - cols * np.sin(np.deg2rad(angle)) - rows * np.cos(np.deg2rad(angle))) / 2

    rotated_image = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            coords = np.dot(M, [j, i, 1])
            x, y = int(coords[0]), int(coords[1])
            if 0 <= x < cols and 0 <= y < rows:
                rotated_image[y, x] = image[i, j]
    augmented_images.append(rotated_image)

    # Salt and pepper noise
    salt_prob = 0.02
    pepper_prob = 0.02
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Add salt (white) noise
    salt_x = np.random.randint(0, image.shape[1], int(num_salt))
    salt_y = np.random.randint(0, image.shape[0], int(num_salt))
    noisy_image[salt_y, salt_x] = 255

    # Add pepper (black) noise
    pepper_x = np.random.randint(0, image.shape[1], int(num_pepper))
    pepper_y = np.random.randint(0, image.shape[0], int(num_pepper))
    noisy_image[pepper_y, pepper_x] = 0

    augmented_images.append(noisy_image)
    
    return augmented_images




def augment_dataset(dataset_path='processed', augmented_path='learning'):
    if not os.path.exists(augmented_path):
        os.makedirs(augmented_path)

    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        augmented_images = augment_image(image)
        for i, aug_img in enumerate(augmented_images):
            aug_img_path = os.path.join(augmented_path, f"{filename.split('.')[0]}_aug_{i}.jpg")
            cv2.imwrite(aug_img_path, aug_img)
            print(f"{aug_img_path} saved.")


def get_video_from_database_and_extract_frames(user_id, video_path, save_path='2fa_check'):
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_capture = [int(total_frames * 0.25), int(total_frames * 0.5), int(total_frames * 0.75)]
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frames_to_capture:
            img_name = f"{save_path}/user_{user_id}_frame_{frame_count}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saved.")
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()



def get_video_from_mongodb(user_id, mongo_uri='mongodb+srv://vankaler:root@paketnikdb.nz3ehj4.mongodb.net/', database_name='PaketnikDB', collection_name='user_videos', save_path='videos'):
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[database_name]
    fs = gridfs.GridFS(db)

    # Query to get the video file from GridFS
    video_file = fs.find_one({'user_id': user_id})
    
    if video_file is None:
        raise ValueError(f"No video found for user_id {user_id}")

    # Ensure the save path directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define the video file path
    video_file_path = os.path.join(save_path, f'user_{user_id}.mp4')

    # Save the video file locally
    with open(video_file_path, 'wb') as f:
        f.write(video_file.read())

    # Close the MongoDB connection
    client.close()

    print(f"Video for user_id {user_id} saved to {video_file_path}")
    return video_file_path



def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Odstranjevanje šuma z Gaussovim zamegljevanjem
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Pretvorba v sivinsko lestvico
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray_image


def send_push_notification(registration_id, message_title, message_body):
    api_key = "9ea96945-3a37-4638-a5d4-22e89fbc998f"
    push_service = FCMNotification(api_key=api_key)
    
    result = push_service.notify_single_device(registration_id=registration_id, message_title=message_title, message_body=message_body)
    print(result)



#registration_id = "DEVICE_REGISTRATION_ID"
#send_push_notification(registration_id, "2FA Verification", "Please verify your login attempt.")
# Zajemanje slik 
capture_video_and_extract_frames(user_id=1)
#preprocess_dataset()
# Augmentacija vseh slik v processed mapi
# augment_dataset()

# video_path = get_video_from_mongodb(user_id=1)  # Replace this with the actual user ID
# get_video_from_database_and_extract_frames(user_id=1, video_path=video_path)