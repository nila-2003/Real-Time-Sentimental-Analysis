# Real-time Sentimental Analysis
<hr>
This repository contains code for a real-time emotion detection system using audio data. The system is built using machine learning techniques, including feature extraction, data augmentation, and a pre-trained neural network model.

## Features
<br>
Real-Time Detection: The system can analyze emotions in real-time from audio input.
Machine Learning Model: A pre-trained neural network model is used for accurate emotion prediction.
Data Augmentation: The training data is augmented to improve model robustness.

Model Loss - 0.464 when trained on 100 epochs

Predicted Emotions:
Anger(1), Disgust(2), Fear(3), Happy(4), Neutral(5), and Sad(6)
## Usage

1. **Preparation:**
   - Install the required dependencies using `pip install -r requirements.txt`.
   - Place your pre-trained model weights (`pretrained_model_weights.h5`) in the project directory.
   - Update the `model_architecture.json` file with the architecture of your pre-trained model.

2. **Run Real-Time Detection:**
   - Use the `real_time_detection` function in the provided Python script to perform real-time emotion detection on audio input.

3. **Customization:**
   - If you want to use your own pre-trained model, make sure to update the model architecture and weights accordingly.
   - Adjust feature extraction, data augmentation, or model parameters as needed.
<hr>
This project utilizes libraries such as Librosa, scikit-learn, and TensorFlow. Special thanks to the contributors of these open-source projects.

Feel free to explore, modify, and integrate this code into your projects. If you encounter issues or have suggestions, please open an issue.


