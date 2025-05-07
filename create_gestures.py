import cv2
import numpy as np
import os
import json
import random
import mediapipe as mp
import imgaug.augmenters as iaa
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Kích thước ảnh
image_x, image_y = 128, 128
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Định nghĩa pipeline augmentation
aug = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10))),
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.Sometimes(0.5, iaa.contrast.LinearContrast((0.75, 1.25)))
])

# Đường dẫn thống nhất
JSON_PATH = "../Code_throw/gestures.json"
GESTURES_DIR = "../Code_throw/gestures"

def init_create_folder():
    """Tạo thư mục gestures nếu chưa tồn tại."""
    try:
        if not os.path.exists(GESTURES_DIR):
            os.makedirs(GESTURES_DIR)
            logging.info(f"Created directory: {GESTURES_DIR}")
    except Exception as e:
        logging.error(f"Failed to create directory: {e}")
        raise

def create_folder(folder_name):
    """Tạo thư mục nếu chưa tồn tại."""
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            logging.info(f"Created directory: {folder_name}")
    except Exception as e:
        logging.error(f"Failed to create directory {folder_name}: {e}")
        raise

def store_in_json(g_id, g_name):
    """Lưu g_id và g_name vào file JSON."""
    try:
        data = {}
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, 'r') as f:
                data = json.load(f)
        if str(g_id) in data:
            choice = input(f"g_id {g_id} already exists with name '{data[str(g_id)]}'. Want to update? (y/n): ")
            if choice.lower() != 'y':
                logging.info("No changes made to JSON")
                return
        data[str(g_id)] = g_name
        with open(JSON_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Stored in JSON: g_id={g_id}, g_name={g_name}")
    except Exception as e:
        logging.error(f"Failed to store in JSON: {e}")
        raise

def is_blurry(image):
    """Kiểm tra ảnh có mờ không."""
    try:
        return cv2.Laplacian(image, cv2.CV_64F).var() < 100
    except Exception as e:
        logging.error(f"Error checking blurry image: {e}")
        return True

def get_hand_region(img):
    """Lấy vùng tay từ ảnh."""
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, y_min, x_max, y_max = img.shape[1], img.shape[0], 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                margin = 20
                x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
                x_max, y_max = min(x_max + margin, img.shape[1]), min(y_max + margin, img.shape[0])
                return img[y_min:y_max, x_min:x_max]
        return None
    except Exception as e:
        logging.error(f"Error getting hand region: {e}")
        return None

def store_images(g_id):
    """Chụp và lưu ảnh cử chỉ."""
    total_pics = 1200 
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        logging.error("Cannot open camera. Please check your webcam.")
        return
    x, y, w, h = 300, 100, 300, 300
    gesture_dir = f"{GESTURES_DIR}/{g_id}"
    create_folder(gesture_dir)
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    logging.info("Starting webcam. Press 'c' to start/stop capturing, 'q' to quit.")
    while True:
        ret, img = cam.read()
        if not ret:
            logging.error("Failed to capture image. Check webcam connection.")
            break
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        # Hiển thị hướng dẫn
        cv2.putText(img, "Press 'c' to start/stop, 'q' to quit", (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (127, 255, 255))
        cv2.putText(img, f"Gesture ID: {g_id}, Images: {pic_no}/{total_pics}", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (127, 255, 255))

        hand_region = get_hand_region(img)
        thresh = np.zeros((image_y, image_x), dtype=np.uint8)
        if hand_region is not None and hand_region.size > 0:
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = [c for c in contours if len(c) >= 3 and c.dtype == np.int32]
            if valid_contours:
                contour = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                x1, y1, w1, h1 = cv2.boundingRect(contour)

                if area > 5000 and 0.5 < w1/h1 < 2 and frames > 10:
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    if is_blurry(save_img):
                        logging.warning("Image blurry, skipping")
                    else:
                        if w1 > h1:
                            save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                        elif h1 > w1:
                            save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
                        save_img = cv2.resize(save_img, (image_x, image_y))

                        # Áp dụng augmentation
                        if random.random() < 0.5:
                            save_img = cv2.flip(save_img, 1)
                        save_img = aug.augment_image(save_img)

                        try:
                            pic_no += 1
                            img_path = f"{gesture_dir}/{pic_no}.jpg"
                            cv2.imwrite(img_path, save_img)
                            logging.info(f"Saved image: {img_path}")
                            cv2.putText(img, "Capturing...", (30, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (127, 255, 255))
                        except Exception as e:
                            logging.error(f"Failed to save image {pic_no}: {e}")
                            pic_no -= 1
                else:
                    logging.debug(f"Invalid contour: area={area}, aspect_ratio={w1/h1 if h1 != 0 else 0}")
            else:
                logging.debug("No valid contours detected")

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)

        if keypress == ord('c'):
            flag_start_capturing = not flag_start_capturing
            frames = 0 if not flag_start_capturing else frames
            logging.info("Capturing started" if flag_start_capturing else "Capturing stopped")
        elif keypress == ord('q'):
            logging.info(f"Stopped capturing. Saved {pic_no} images for gesture {g_id}")
            break
        if flag_start_capturing:
            frames += 1
        if pic_no >= total_pics:
            logging.info(f"Completed capturing {total_pics} images for gesture {g_id}")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        init_create_folder()
        print("Enter a unique gesture number (e.g., 1, 2, 3):")
        g_id = input("Enter gesture no.: ").strip()
        if not g_id.isdigit():
            raise ValueError("Gesture number must be an integer.")
        g_id = int(g_id)
        print("Enter gesture name (e.g., 'one', 'hello'):")
        g_name = input("Enter gesture name/text: ").strip()
        if not g_name:
            raise ValueError("Gesture name cannot be empty.")
        store_in_json(g_id, g_name)
        store_images(g_id)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")
    except ValueError as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")