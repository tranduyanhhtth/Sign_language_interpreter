import cv2
import os
import numpy as np
from glob import glob
from sklearn.utils import shuffle
import pickle
import logging

# Read images from the "gestures" directory 

# => divide them into training, testing, and validation sets
# 	Train: 5/6 of the data
# 	Test: 1/12 of the data
# 	Validation: 1/12 of the data

# => save them as pickle files.
#   train_images, train_labels
#   test_images, test_labels
#   val_images, val_labels

# => pickle files are used to cnn_model.py to train the CNN model 

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pickle_images_labels():
    """Đọc ảnh từ thư mục gestures và trả về danh sách (ảnh, nhãn)."""
    images_labels = []
    images = glob("../Code_throw/gestures/*/*.jpg")
    images.sort()
    for image in images:
        try:
            # Trích xuất g_id từ tên thư mục (ví dụ: '1' từ '../Code_throw/gestures/1/1.jpg')
            label = os.path.basename(os.path.dirname(image))
            logging.debug(f"Processing image: {image}, label: {label}")
            
            # Kiểm tra label là số nguyên hợp lệ
            try:
                int_label = int(label)
            except ValueError:
                logging.error(f"Invalid label '{label}' in path {image}. Skipping.")
                continue
            
            # Đọc ảnh ở định dạng grayscale
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.error(f"Failed to read image {image}. Skipping.")
                continue
            
            # Kiểm tra kích thước ảnh
            if img.shape != (128, 128):
                logging.warning(f"Image {image} has shape {img.shape}, expected (128, 128). Resizing.")
                img = cv2.resize(img, (128, 128))
            
            images_labels.append((np.array(img, dtype=np.uint8), int_label))
        except Exception as e:
            logging.error(f"Error processing image {image}: {e}")
            continue
    
    logging.info(f"Loaded {len(images_labels)} valid images")
    return images_labels

def main():
    """Chia dữ liệu thành tập train, test, val và lưu thành file pickle."""
    try:
        # Đọc và trộn dữ liệu
        images_labels = pickle_images_labels()
        if not images_labels:
            raise ValueError("No valid images loaded.")
        
        # Trộn dữ liệu với random seed cố định
        images_labels = shuffle(images_labels, random_state=42)
        images, labels = zip(*images_labels)
        total_images = len(images)
        logging.info(f"Total images: {total_images}")

        # Chia dữ liệu: 80% train, 10% test, 10% val
        train_end = int(0.8 * total_images)
        test_end = int(0.9 * total_images)

        # Train
        train_images = images[:train_end]
        train_labels = labels[:train_end]
        logging.info(f"Train set: {len(train_images)} images")
        with open("../Code_throw/train_images", "wb") as f:
            pickle.dump(train_images, f)
        with open("../Code_throw/train_labels", "wb") as f:
            pickle.dump(train_labels, f)

        # Test
        test_images = images[train_end:test_end]
        test_labels = labels[train_end:test_end]
        logging.info(f"Test set: {len(test_images)} images")
        with open("../Code_throw/test_images", "wb") as f:
            pickle.dump(test_images, f)
        with open("../Code_throw/test_labels", "wb") as f:
            pickle.dump(test_labels, f)

        # Validation
        val_images = images[test_end:]
        val_labels = labels[test_end:]
        logging.info(f"Validation set: {len(val_images)} images")
        with open("../Code_throw/val_images", "wb") as f:
            pickle.dump(val_images, f)
        with open("../Code_throw/val_labels", "wb") as f:
            pickle.dump(val_labels, f)

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()