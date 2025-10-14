from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have true labels and predicted labels
y_true = np.array([4, 3, 4, 2, 1, 4, 3, 5, 3, 1])  # Replace with actual class labels
y_pred = np.array([3, 1, 2, 5, 6, 7, 3, 4, 6, 3])  # Replace with predicted class labels

# Class labels for the skin cancer types
class_labels = [
    "AKIEC",  # Actinic keratoses and intraepithelial carcinoma
    "BCC",    # Basal cell carcinoma
    "BKL",    # Benign keratosis-like lesions
    "DF",     # Dermatofibroma
    "MEL",    # Melanoma
    "NV",     # Melanocytic nevi
    "VASC"    # Vascular lesions
]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
plt.title("Confusion Matrix for Skin Cancer Classification")
plt.show()
