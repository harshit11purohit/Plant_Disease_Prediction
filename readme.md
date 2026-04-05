🌱 Plant Disease Classifier
An End-to-End Deep Learning Solution for Agricultural Health Monitoring

📌 Project Overview
This project leverages Convolutional Neural Networks (CNN) to identify 38 different classes of plant diseases across various species including Tomato, Potato, Grape, and Corn. It features a high-performance training pipeline, a fine-tuning script for custom data, and a user-friendly Streamlit web interface for real-time diagnosis.

Key Features:
Deep Learning Model: Built with TensorFlow/Keras using a custom CNN architecture.

Transfer Learning: Support for fine-tuning on custom "Google Images" datasets.

Compatibility: Optimized for Python 3.13 using manual weight rebuilding to bypass Keras versioning bugs.

Web Dashboard: Instant image upload and classification via Streamlit.

📂 Project Structure
Plaintext
PLANT_DISEASE_PREDICTION/
├── app/
│   ├── images/                # UI Assets (leaf icons, etc.)
│   ├── main.py                # Streamlit Web Application
│   ├── fine_tune_main.py      # Transfer Learning Script
│   ├── class_indices.json     # Class Mapping (0-37)
│   └── requirements.txt       # Dependencies
├── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb  # Training Notebook
└── plant_disease_model2_v1.h5 # Pre-trained Model Weights
🚀 Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/harshit11purohit/Plant-Disease-Prediction.git
cd plant-disease-prediction
2. Install Dependencies
Note: Optimized for Python 3.13 environments.

Bash
pip install numpy streamlit tensorflow pillow
3. Run the Application
Bash
cd app
streamlit run main.py
🧪 Training & Fine-Tuning
Base Training
The model is trained on the PlantVillage Dataset. To retrain from scratch, run the Jupyter Notebook:
Plant_Disease_Prediction_CNN_Image_Classifier.ipynb

Fine-Tuning (Custom Data)
To update the model with new images (e.g., from Google Images):

Place images in app/google_images/ (categorized by subfolders).

Run the fine-tuning script:

Bash
python app/fine_tune_main.py
📊 Performance Metrics
Total Classes: 38

Input Image Size: 224 x 224

Training Accuracy: ~90%+ (Final Version)

Inference Speed: < 2s per image

🛠️ Troubleshooting (Python 3.13)
If you encounter TypeError: Unrecognized keyword arguments: {'quantization_config': None}:

Solution: The project uses a Manual Rebuild method in main.py which bypasses metadata errors by defining the architecture in code and loading weights separately via model.load_weights().

Author:Harshit Purohit
