# AssistNet
AssistNet is a Computer Vision-based tool designed to assist aircraft navigation by classifying taxiways and runways using Convolutional Neural Networks (CNNs) that leverages the AssistTaxi Dataset. 

![Python](https://img.shields.io/badge/Python-3.7.17-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## Table of Contents
01. [Features](#features)
02. [Installation](#installation)
03. [Usage](#usage)
04. [Dataset Preparation](#dataset-preparation)
05. [Model Architecture](#model-architecture)
06. [Results](#results)
07. [Future Work](#future-work)
08. [License](#license)
09. [Contact](#contact)

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

## Results
<img src="https://github.com/user-attachments/assets/a4baeb40-69e3-46e6-abca-6ce7f1feb2d5" width="700" height="700" alt="Confusion Matrix Sample 1">
<img src="https://github.com/user-attachments/assets/724c10e4-51ed-4000-aad4-367dcad77821" width="700" height="700" alt="Confusion Matrix Sample 2">

### Training Metrics
Below are the training and validation metrics tracked during the model's training process:

<img src="https://github.com/user-attachments/assets/3bb3c8bd-1a24-4bf1-9f9a-48090435825e" width="500" height="300" alt="Confusion Matrix Sample 1">

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


