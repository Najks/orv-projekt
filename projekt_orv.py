import cv2
import os

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

# Zajemanje slik za uporabnika s ID 1
capture_video_and_extract_frames(user_id=1)
