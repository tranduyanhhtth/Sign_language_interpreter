import cv2
import numpy as np
import tensorflow as tf
import os
import json
import pyttsx3
from tensorflow.keras.models import load_model
from threading import Thread
import mediapipe as mp
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('mediapipe').setLevel(logging.WARNING)

# Tắt GPU để tránh cảnh báo
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Khởi tạo biến toàn cục
is_voice_on = True

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Khởi tạo pyttsx3
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    logging.info("pyttsx3 initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize pyttsx3: {e}")
    raise

# Tải và biên dịch mô hình Keras
try:
    model = load_model('../Code_throw/cnn_model_keras2.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info("Model loaded and compiled successfully")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# Kích thước ảnh cố định
image_x, image_y = 128, 128

# Đường dẫn JSON
JSON_PATH = "../Code_throw/gestures.json"

def keras_process_image(img):
    """Chuẩn bị ảnh cho dự đoán Keras."""
    try:
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.reshape(img, (1, image_x, image_y, 1))
        return img
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None

def keras_predict(model, image):
    """Dự đoán lớp và xác suất từ ảnh."""
    processed = keras_process_image(image)
    if processed is None:
        return 0.0, 0
    try:
        pred_probab = model.predict(processed, verbose=0)[0]
        pred_class = np.argmax(pred_probab)
        logging.info(f"Prediction: prob={pred_probab[pred_class]*100:.2f}, class={pred_class}")
        return pred_probab[pred_class], pred_class
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return 0.0, 0

def get_pred_text_from_json(pred_class):
    """Lấy tên cử chỉ từ file JSON."""
    try:
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, 'r') as f:
                data = json.load(f)
            g_id = str(pred_class)
            g_name = data.get(g_id, "")
            if g_name:
                logging.info(f"Found gesture: g_id={g_id}, g_name={g_name}")
            else:
                logging.warning(f"No gesture found for g_id={g_id}")
            return g_name
        else:
            logging.error(f"JSON file not found: {JSON_PATH}")
            return ""
    except Exception as e:
        logging.error(f"JSON error: {e}")
        return ""

def get_pred_from_contour(contour, thresh):
    """Dự đoán số từ contour."""
    try:
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        save_img = thresh[y1:y1+h1, x1:x1+w1]
        if save_img.size == 0 or w1 < 10 or h1 < 10:
            logging.debug("Invalid contour image size")
            return "", 0.0
        save_img = cv2.resize(save_img, (image_x, image_y))
        pred_probab, pred_class = keras_predict(model, save_img)
        if pred_probab * 100 > 40:
            pred_text = get_pred_text_from_json(pred_class)
            # Hiển thị bất kỳ g_name nào từ JSON
            if pred_text:
                return pred_text, pred_probab * 100
        logging.debug(f"Prediction below threshold or not a number: {pred_probab*100:.2f}%")
        return "", pred_probab * 100
    except Exception as e:
        logging.error(f"Error in get_pred_from_contour: {e}")
        return "", 0.0

def get_img_contour_thresh(img):
    """Phát hiện vùng tay và trả về contours, thresh."""
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
                margin = 30
                x_min, y_min = max(x_min - margin, 0), max(y_min - margin, 0)
                x_max, y_max = min(x_max + margin, img.shape[1]), min(y_max + margin, img.shape[0])
                roi = img[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    return img, contours, thresh
        logging.debug("No hand detected")
        return img, [], np.zeros((image_y, image_x), dtype=np.uint8)
    except Exception as e:
        logging.error(f"Error in get_img_contour_thresh: {e}")
        return img, [], np.zeros((image_y, image_x), dtype=np.uint8)

def say_text(text):
    """Phát âm văn bản bằng pyttsx3."""
    global is_voice_on
    if not is_voice_on or not text:
        return
    try:
        engine.say(text)
        engine.runAndWait()
        logging.info(f"Played audio: {text}")
    except Exception as e:
        logging.error(f"Error in say_text: {e}")

def recognize():
    """Chạy nhận diện số từ 0-10."""
    global is_voice_on
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        logging.error("Cannot open webcam")
        return

    pred_text = ""
    count_same_frames = 0
    confidence = 0.0
    contour_area = 0.0

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                logging.error("Failed to capture image")
                break
            # Lật hình ảnh ngang (sửa hiệu ứng gương)
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (320, 240))
            img, contours, thresh = get_img_contour_thresh(img)
            old_pred_text = pred_text

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(contour)
                if contour_area > 1000:
                    pred_text, confidence = get_pred_from_contour(contour, thresh)
                    if old_pred_text == pred_text and pred_text:
                        count_same_frames += 1
                    else:
                        count_same_frames = 0

                    if count_same_frames > 5 and pred_text:
                        Thread(target=say_text, args=(pred_text,)).start()
                        count_same_frames = 0
            else:
                contour_area = 0.0
                confidence = 0.0
                pred_text = ""

            blackboard = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blackboard, "Number Recognition", (50, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 0, 0))
            cv2.putText(blackboard, "Predicted: " + pred_text, (15, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0))
            cv2.putText(blackboard, f"Confidence: {confidence:.2f}%", (15, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0))
            cv2.putText(blackboard, f"Contour Area: {contour_area:.2f}", (15, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0))
            cv2.putText(blackboard, "Voice ON" if is_voice_on else "Voice OFF", (225, 220), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 127, 0))
            cv2.putText(blackboard, "Instructions:", (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
            cv2.putText(blackboard, "q: Quit, v: Toggle Voice", (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
            cv2.putText(blackboard, "Show hand to recognize 0-10", (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

            cv2.rectangle(img, (150, 50), (300, 200), (0, 255, 0), 2)
            res = np.hstack((img, blackboard))
            cv2.imshow("Recognizing gesture", res)
            cv2.imshow("thresh", thresh)
            keypress = cv2.waitKey(1)
            if keypress == ord('q'):
                break
            if keypress == ord('v'):
                is_voice_on = not is_voice_on
                logging.info(f"Voice status: {'ON' if is_voice_on else 'OFF'}")

    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        for path in ['../Code_throw/gestures/1/100.jpg', '../Code_throw/gestures.json', '../Code_throw/cnn_model_keras2.h5']:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}")
        test_img = np.zeros((image_x, image_y), dtype=np.uint8)
        keras_predict(model, test_img)
        recognize()
    except Exception as e:
        logging.error(f"Main execution error: {e}")