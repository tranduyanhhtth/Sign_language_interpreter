import cv2
import os

def flip_images():
    gest_folder = "../Code_throw/gestures"
    
    # Verify if the gestures folder exists
    if not os.path.exists(gest_folder):
        print(f"Error: Directory '{gest_folder}' does not exist.")
        return
    
    # Iterate through each gesture folder (e.g., '0', '1', etc.)
    for g_id in os.listdir(gest_folder):
        gesture_path = os.path.join(gest_folder, g_id)
        
        # Check if it's a directory
        if not os.path.isdir(gesture_path):
            print(f"Skipping '{gesture_path}' (not a directory)")
            continue
        
        # Get list of image files in the gesture folder
        image_files = [f for f in os.listdir(gesture_path) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} images in '{gesture_path}'")
        
        # Process each image
        for i, img_name in enumerate(image_files):
            path = os.path.join(gesture_path, img_name)
            # Generate new filename for flipped image (e.g., 1201.jpg for 1.jpg)
            new_idx = i + 1 + len(image_files)
            new_path = os.path.join(gesture_path, f"{new_idx}.jpg")
            
            try:
                # Read the image in grayscale
                img = cv2.imread(path, 0)
                if img is None:
                    print(f"Error: Failed to load '{path}'")
                    continue
                
                # Flip the image horizontally
                img_flipped = cv2.flip(img, 1)
                
                # Save the flipped image
                cv2.imwrite(new_path, img_flipped)
                print(f"Flipped and saved: '{new_path}'")
                
            except Exception as e:
                print(f"Error processing '{path}': {str(e)}")
                continue

if __name__ == "__main__":
    flip_images()