import matplotlib.pyplot as plt

# Data for plotting
epochs = list(range(1, 11))  # 10 epochs
accuracy = [0.5401, 0.7150, 0.7572, 0.7859, 0.8027, 0.8126, 0.8251, 0.8351, 0.8441, 0.8509]
val_accuracy = [0.67693, 0.6539, 0.69315, 0.7104, 0.7009, 0.72681, 0.6952, 0.7188, 0.6771, 0.6946]
loss = [1.3610, 0.8210, 0.7114, 0.6465, 0.6050, 0.5767, 0.5484, 0.5225, 0.4969, 0.4786]
val_loss = [0.8492, 1.0256, 0.9979, 0.9279, 1.1615, 1.0556, 1.9458, 1.3065, 1.6498, 1.6023]

# Create subplots
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training Loss', marker='o', color='red')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='orange')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Show the plots
plt.tight_layout()
plt.show()
