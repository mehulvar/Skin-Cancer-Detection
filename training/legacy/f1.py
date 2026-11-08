import numpy as np
import matplotlib.pyplot as plt

cm = np.array([
    [1793, 315, 263,  97, 218, 147,  83],
    [ 258, 1007, 197,  73, 147, 104,  53],
    [ 205, 176, 1708,  98, 248, 193,  92],
    [ 103,  64,  78, 811, 102,  69,  43],
    [ 297, 153, 206, 104, 2514, 307, 148],
    [ 203, 103, 148,  82, 198, 1706, 104],
    [  81,  53,  87,  42, 104,  83, 913]
])

class_labels = ['FINALvascularlesion', 'FINALdermatofibroma', 'FINALbasalcell', 'FINALactinickeratosis', 'FINALmelanoma', 'FINALnevus', 'FINALbenignkeratosis']
f1_scores = []

for i in range(len(class_labels)):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

# Create the bar graph
plt.figure(figsize=(10, 6))
plt.bar(class_labels, f1_scores, color='skyblue')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.title('F1-Scores for Each Skin Lesion Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nF1-Scores for Each Class:")
for i, label in enumerate(class_labels):
    print(f"{label}: {f1_scores[i]:.4f}")