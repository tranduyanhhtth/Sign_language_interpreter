import numpy as np
import pickle
import cv2
import os
import logging
import json
from glob import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    """Trả về kích thước ảnh từ mẫu hoặc mặc định 128x128."""
    sample_path = '../Code_throw/gestures/1/100.jpg'
    try:
        img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh mẫu {sample_path}")
        logging.info(f"Image size from sample: {img.shape}")
        return img.shape
    except Exception as e:
        logging.warning(f"Lỗi khi đọc ảnh mẫu: {e}. Sử dụng kích thước mặc định 128x128")
        return (128, 128)

def get_gesture_mapping():
    """Đọc ánh xạ g_id -> g_name từ gestures.json và tạo ánh xạ nhãn."""
    json_path = '../Code_throw/gestures.json'
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Không tìm thấy file {json_path}. Chạy create_gestures.py để tạo.")
        with open(json_path, 'r') as f:
            gesture_dict = json.load(f)
        # Tạo ánh xạ g_id (str) -> index (int, bắt đầu từ 0)
        unique_g_ids = sorted([int(g_id) for g_id in gesture_dict.keys()])
        label_mapping = {g_id: idx for idx, g_id in enumerate(unique_g_ids)}
        reverse_mapping = {idx: g_id for g_id, idx in label_mapping.items()}
        logging.info(f"Gesture mapping: {label_mapping}")
        return label_mapping, reverse_mapping, len(unique_g_ids)
    except Exception as e:
        logging.error(f"Lỗi khi đọc gestures.json: {e}")
        raise

def validate_gestures():
    """Kiểm tra sự đồng bộ giữa gestures.json và thư mục gestures/."""
    try:
        label_mapping, _, _ = get_gesture_mapping()
        gesture_dirs = [int(os.path.basename(d)) for d in glob('../Code_throw/gestures/*') if os.path.isdir(d)]
        missing_in_json = [g_id for g_id in gesture_dirs if g_id not in label_mapping]
        missing_in_dirs = [g_id for g_id in label_mapping if g_id not in gesture_dirs]
        if missing_in_json:
            logging.warning(f"Các g_id có trong gestures/ nhưng thiếu trong gestures.json: {missing_in_json}")
            logging.warning("Cập nhật gestures.json bằng create_gestures.py hoặc xóa thư mục thừa.")
        if missing_in_dirs:
            logging.warning(f"Các g_id có trong gestures.json nhưng thiếu thư mục gestures/: {missing_in_dirs}")
            logging.warning("Tạo lại thư mục bằng create_gestures.py hoặc xóa mục trong gestures.json.")
        return missing_in_json, missing_in_dirs
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra gestures: {e}")
        raise

def get_num_of_classes():
    """Trả về số lớp từ gestures.json."""
    try:
        _, _, num_classes = get_gesture_mapping()
        if num_classes == 0:
            raise ValueError("Không tìm thấy lớp nào trong gestures.json")
        logging.info(f"Số lớp tìm thấy: {num_classes}")
        return num_classes
    except Exception as e:
        logging.error(f"Lỗi khi đếm số lớp: {e}")
        raise

image_x, image_y = get_image_size()

def cnn_model():
    """Tạo mô hình CNN cho phân loại đa lớp."""
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Input(shape=(image_x, image_y, 1)))
    model.add(Conv2D(16, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = SGD(learning_rate=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "../Code_throw/cnn_model_keras2.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    return model, callbacks_list

def normalize_labels(labels, label_mapping):
    """Chuyển g_id thành nhãn bắt đầu từ 0."""
    try:
        normalized_labels = np.array([label_mapping.get(label, -1) for label in labels], dtype=np.int32)
        if -1 in normalized_labels:
            invalid_labels = sorted(list(set([label for label, norm_label in zip(labels, normalized_labels) if norm_label == -1])))
            raise ValueError(
                f"Nhãn không hợp lệ trong dữ liệu: {invalid_labels}. "
                "Kiểm tra gestures.json có chứa các g_id này không. "
                "Chạy create_gestures.py để thêm gesture hoặc chạy load_images.py để cập nhật pickle files."
            )
        return normalized_labels
    except Exception as e:
        logging.error(f"Lỗi khi chuẩn hóa nhãn: {e}")
        raise

def train():
    """Huấn luyện mô hình CNN."""
    try:
        # Kiểm tra sự đồng bộ giữa gestures.json và gestures/
        missing_in_json, missing_in_dirs = validate_gestures()

        # Đọc dữ liệu từ file pickle
        with open("../Code_throw/train_images", "rb") as f:
            train_images = np.array(pickle.load(f))
        with open("../Code_throw/train_labels", "rb") as f:
            train_labels = np.array(pickle.load(f), dtype=np.int32)
        with open("../Code_throw/val_images", "rb") as f:
            val_images = np.array(pickle.load(f))
        with open("../Code_throw/val_labels", "rb") as f:
            val_labels = np.array(pickle.load(f), dtype=np.int32)

        # Ghi log thông tin dữ liệu
        logging.info(f"Train images shape: {train_images.shape}, labels shape: {train_labels.shape}")
        logging.info(f"Validation images shape: {val_images.shape}, labels shape: {val_labels.shape}")
        logging.info(f"Unique train labels: {np.unique(train_labels)}")
        logging.info(f"Unique validation labels: {np.unique(val_labels)}")

        # Chuẩn hóa nhãn
        label_mapping, reverse_mapping, num_of_classes = get_gesture_mapping()
        train_labels = normalize_labels(train_labels, label_mapping)
        val_labels = normalize_labels(val_labels, label_mapping)

        # Reshape ảnh
        train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
        val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))

        # Chuyển nhãn sang dạng one-hot
        train_labels = to_categorical(train_labels, num_classes=num_of_classes)
        val_labels = to_categorical(val_labels, num_classes=num_of_classes)

        logging.info(f"After reshape: Train images shape: {train_images.shape}, labels shape: {train_labels.shape}")
        logging.info(f"After reshape: Validation images shape: {val_images.shape}, labels shape: {val_labels.shape}")

        # Tạo và huấn luyện mô hình
        model, callbacks_list = cnn_model()
        model.summary()
        model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            epochs=20,
            batch_size=32,
            callbacks=callbacks_list
        )

        # Đánh giá mô hình
        scores = model.evaluate(val_images, val_labels, verbose=0)
        logging.info(f"CNN Accuracy: {scores[1]*100:.2f}%")
        logging.info(f"CNN Error: {100 - scores[1]*100:.2f}%")

    except Exception as e:
        logging.error(f"Lỗi trong quá trình huấn luyện: {e}")
        raise

if __name__ == "__main__":
    train()