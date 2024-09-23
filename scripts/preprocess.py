import cv2
import os

# Define input directories
input_dirs = [
    "/workspaces/Gujrati-learning-application/gujarati_character_recognition/images/archive/Gujarati OCR",
    "/workspaces/Gujrati-learning-application/gujarati_character_recognition/images/archive/Joint_Characters",
    "/workspaces/Gujrati-learning-application/gujarati_character_recognition/images/archive/TeraFont_Varun"
]

# Create a folder to save preprocessed images
preprocessed_dir = "../Gujrati-learning-application/gujarati_character_recognition/data/preprocessed"
os.makedirs(preprocessed_dir, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)  # Read the image in grayscale
    
    # Resize image (optional, adjust size as needed)
    img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
    
    # Denoising
    denoised_img = cv2.fastNlMeansDenoising(img, h=30)
    
    # Thresholding to get binary image
    _, binary_img = cv2.threshold(denoised_img, 128, 255, cv2.THRESH_BINARY)
    
    # Save preprocessed image
    filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(preprocessed_dir, filename), binary_img)
    
    return binary_img

# Preprocess all images in the input directories and their subdirectories
for input_dir in input_dirs:
    for root, dirs, files in os.walk(input_dir):
        for img_name in files:
            if img_name.endswith(".jpeg") or img_name.endswith(".jpg") or img_name.endswith(".png"):
                img_path = os.path.join(root, img_name)
                print(f"Processing {img_path}...")
                preprocess_image(img_path)

print(f"Preprocessing complete. Preprocessed images saved to {preprocessed_dir}.")
