Skin Cancer Project

Overview
This project uses a Convolutional Neural Network (CNN) to detect and classify different types of skin cancer from dermatoscopic images.  
It was developed as part of a research project exploring how artificial intelligence can assist in early skin cancer detection and diagnosis.

Dataset
The model is trained on the HAM10000 dataset and additional images downloaded from the ISIC Archive.  
The dataset includes seven classes of skin lesions:
- Melanocytic nevi (nv)
- Melanoma (mel)
- Benign keratosis-like lesions (bkl)
- Basal cell carcinoma (bcc)
- Actinic keratoses (akiec)
- Vascular lesions (vasc)
- Dermatofibroma (df)

Model Architecture
The CNN includes:
- Convolutional layers for feature extraction  
- Max pooling for dimensionality reduction  
- Fully connected layers for classification  
- Softmax activation for output probabilities  

Data augmentation and dropout were used to reduce overfitting.

Tools and Libraries
- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- scikit-learn
- Google Colab
- Visual Studio Code

How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/mehulvar/skin-cancer-project.git
