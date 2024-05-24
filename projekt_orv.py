import cv2
import os
import numpy as np


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


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Odstranjevanje šuma z Gaussovim zamegljevanjem
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Pretvorba v sivinsko lestvico
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray_image


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
    
    # Horizontalna zrcalna slika
    flip_horizontal = cv2.flip(image, 1)
    augmented_images.append(flip_horizontal)
    
    # Povečanje svetlosti
    bright_image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(bright_image)
    
    # Povečanje kontrasta
    contrast_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    augmented_images.append(contrast_image)
    
    # Rotacija
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    augmented_images.append(rotated_image)
    
    return augmented_images


def augment_dataset(dataset_path='processed', augmented_path='augmented'):
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



# Zajemanje slik 
#capture_video_and_extract_frames(user_id=1)
#preprocess_dataset()
# Augmentacija vseh slik v processed mapi
#augment_dataset()