# AssistNet
AssistNet is a Computer Vision-based tool designed to assist aircraft navigation by classifying taxiways and runways using Convolutional Neural Networks (CNNs) that leverages the AssistTaxi Dataset. 

![Python](https://img.shields.io/badge/Python-3.7.17-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## Table of Contents
01. [Features](#features)
02. [Installation](#installation)
03. [Usage](#usage)
04. [Model Architecture](#model-architecture)
05. [Results](#results)
06. [Future Work](#future-work)
07. [License](#license)
08. [Contact](#contact)

## Features
- Classifies taxiways and runways with high accuracy.
- Supports Piper aircraft images.
- Integrates video frame splicing for enhanced dataset preparation.
- Works on Python 3.7+ and uses TensorFlow/Keras for deep learning.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/assist-taxi.git
   cd assist-taxi
2. Create and activate a virtual environment:
    ```bash
    python3.7 -m venv env
    source env/bin/activate  # using Mac/Linux
    .\env\Scripts\activate   # using Windows
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. Verify the installation:
   ```bash
   python -c "import tensorflow; print(tensorflow.__version__)"


## Usage
### 1. Data Preparation
   - Organize your dataset as follows to separate training, validation, and testing data. Each folder should contain subfolders for taxiway and runway images:
     ```
     C:/Users/your_username/AssistNet/directory/
     ├── training/
     │   ├── taxiway/
     │   └── runway/
     ├── validation/
     │   ├── taxiway/
     │   └── runway/
     ├── testing/
     │   ├── taxiway/
     │   └── runway/
     ```
   - Place the **taxiway** and **runway** images in their respective folders under training, validation, and testing.
   - Each subfolder should contain images labeled according to the category they represent (e.g., t_00001.jpg for taxiways and r_00001.jpg for runways).
#### Video Frame Splicing (Optional)
To enhance your dataset, you can extract frames from video recordings of taxiways and runways. This is particularly useful for capturing various perspectives and lighting conditions. Here’s how to prepare frames from video files:

**Video Splicing:** Use a tool or script to split video files into individual frames.
Extract frames at regular intervals (e.g., every 5th or 10th frame) to avoid redundant images.
Save each frame in the appropriate folder (taxiway or runway) depending on the video content.

**Image Cropping:** If your images contain extraneous elements or backgrounds, consider cropping them to focus on the runway or taxiway sections. This can improve classification accuracy.

**Resolution and Resizing:** Standardize the resolution of your images to match the input size expected by the model (e.g., 224x224 or 404x225). Consistent image dimensions improve the model’s performance and reduce computational complexity.

### 2. Balancing the Dataset
Ensure that both classes (taxiway and runway) are balanced within each dataset split (training, validation, testing). An imbalanced dataset can lead to biased model performance, where the model favors the class with more examples. If you encounter an imbalance:

Use oversampling or data augmentation on the minority class to create a balanced dataset.
Adjust training strategies if imbalances are unavoidable.

###  Training the Model
   - Run the following command to train the model:
###  Evaluating the Model
   - Once training is complete, evaluate the model on the test set:
     ```bash
     python evaluate.py --data_path C:/Users/your_username/AssistNet/directory/testing/
     ```
   - This will generate a confusion matrix and additional evaluation metrics like accuracy and precision saved to your output directory.




## Model Architecture
<img src="https://github.com/user-attachments/assets/007296c3-4712-4297-8d06-e8af831dcc19" width="600" height="300" alt="Confusion Matrix Sample 2">

AssistNet employs a CNN architecture designed for high accuracy in image classification tasks. Key features include convolutional, pooling, and fully connected layers optimized for distinguishing between taxiway and runway images.

## Results
<img src="https://github.com/user-attachments/assets/588e6ddd-8817-4ec6-b1b0-c1a931acd86c" width="700" height="700" alt="Confusion Matrix Sample 2">

### Training Metrics
Below are the training and validation metrics tracked during the model's training process:

<img src="https://github.com/user-attachments/assets/40cacc6a-9f02-40c4-8104-6a27e4fe6087" width="600" height="500"> 

- **Accuracy**: Consistent improvement was observed as the epochs progressed, indicating effective learning.
- **Loss**: The model's loss steadily decreased, confirming convergence and reduced error in predictions.



## Future Work
- Implement real-time classification using video input.
- Extend the model to classify additional aircraft types.
- Integrate with aircraft systems for seamless assistance.

## Contact
For questions, suggestions, or collaborations:
- **Amy Alvarez** - alvareza2023@my.fit.edu
- **Parth Ganeriwala** - pganeriwala2022@my.fit.edu
- [GitHub Profile](https://github.com/your-username)


