import cv2
import numpy as np
import os

# function to remove black borders from the image
def remove_black_borders(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find all the non-black pixels (content area)
    coords = cv2.findNonZero(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1])

    if coords is not None:
        # Get the bounding box of the non-black area
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y + h, x:x + w]  # Crop the black borders
    else:
        return image  # If no non-black area is detected, return the original image

# Function to crop or resize an image to 400x224 aspect ratio and remove black borders
def crop_image(image_path, output_path):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Remove black borders if they exist
    cropped_image = remove_black_borders(image)

    # Get new dimensions after removing black borders
    h, w, _ = cropped_image.shape

    # Desired crop dimensions (maintain 16:9 aspect ratio)
    crop_w, crop_h = 400, 225

    # Check if the image has black borders and adjust the crop size accordingly
    if np.array_equal(cropped_image, image) == False:
        print(f"Black borders detected, applying 400x224 crop for: {image_path}")

    if w >= crop_w and h >= crop_h:
        # Center cropping: Centered both horizontally and vertically for 16:9 aspect ratio
        x_start = (w - crop_w) // 2
        y_start = (h - crop_h) // 2
        final_cropped_image = cropped_image[y_start:y_start + crop_h, x_start:x_start + crop_w]
    else:
        # Resize if the image is smaller than the desired crop size
        print(f"Image {image_path} is too small, resizing instead of cropping.")
        final_cropped_image = cv2.resize(cropped_image, (crop_w, crop_h))

    # Save the cropped or resized image
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    cv2.imwrite(output_path, final_cropped_image)
    print(f"Saved cropped image: {output_path}")

# Function to apply cropping to an entire dataset
def apply_cropping_to_dataset(input_directory, output_directory):
    print(f"Processing dataset from {input_directory} to {output_directory}")
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_directory, file)
                crop_image(input_path, output_path)

# Apply cropping to your dataset paths
apply_cropping_to_dataset('C:/Users/amyba/OneDrive/Desktop/ML_CLASSIFY/directory/training',
                          'C:/Users/amyba/OneDrive/Desktop/ML_CLASSIFY/directory/training_cropped')

apply_cropping_to_dataset('C:/Users/amyba/OneDrive/Desktop/ML_CLASSIFY/directory/validation',
                          'C:/Users/amyba/OneDrive/Desktop/ML_CLASSIFY/directory/validation_cropped')
