import os
from sklearn.model_selection import train_test_split
import shutil

# Path for the original dataset (where all images are stored)
original_data_dir = 'C:/Users/amyba/OneDrive/Desktop/ML_CLASSIFY2/directory/original_data_dir' #where all data is kept
train_dir = 'C:/Users/amyba/OneDrive/Desktop/ML_CLASSIFY2/directory/final_training'
val_dir = 'C:/Users/amyba/OneDrive/Desktop/ML_CLASSIFY2/directory/final_validation'
test_dir = 'C:/Users/amyba/OneDrive/Desktop/ML_CLASSIFY2/directory/final_testing'

# ensure directories for train, validation, and test are created
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to copy images to respective directories
def copy_files(file_list, src_folder, dest_folder):
    for file_name in file_list:
        shutil.copy(os.path.join(src_folder, file_name), os.path.join(dest_folder, file_name))

# Function to split data based on image labels
def split_data():
    # Get all images in the original data directory
    images = os.listdir(original_data_dir)
    
    # Separate images into taxiway and runway based on filename prefix
    taxiway_images = [f for f in images if f.lower().startswith('t_')]
    runway_images = [f for f in images if f.lower().startswith('r_')]

    # Split taxiway images: 70% train, 20% validation, 10% test
    train_taxiway, temp_taxiway = train_test_split(taxiway_images, test_size=0.3, random_state=42)
    val_taxiway, test_taxiway = train_test_split(temp_taxiway, test_size=0.333, random_state=42)

    # Split runway images: 70% train, 20% validation, 10% test
    train_runway, temp_runway = train_test_split(runway_images, test_size=0.3, random_state=42)
    val_runway, test_runway = train_test_split(temp_runway, test_size=0.333, random_state=42)

    # Copy taxiway images to their respective directories
    copy_files(train_taxiway, original_data_dir, os.path.join(train_dir, 'taxiway'))
    copy_files(val_taxiway, original_data_dir, os.path.join(val_dir, 'taxiway'))
    copy_files(test_taxiway, original_data_dir, os.path.join(test_dir, 'taxiway'))

    # Copy runway images to their respective directories
    copy_files(train_runway, original_data_dir, os.path.join(train_dir, 'runway'))
    copy_files(val_runway, original_data_dir, os.path.join(val_dir, 'runway'))
    copy_files(test_runway, original_data_dir, os.path.join(test_dir, 'runway'))

# Run the split
split_data()
