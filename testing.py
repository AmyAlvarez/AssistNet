# Import necessary libraries
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

# defining the test data directory
test_directory = 'directory/final_testing'

# loads the trained model
model = tf.keras.models.load_model('directory/taxiway_v_runway_model.h5')

# Create an ImageDataGenerator for test data (rescale pixel values)
test_data_generator = ImageDataGenerator(rescale=1.0 / 255)

# loads the test data using flow_from_directory
test_generator = test_data_generator.flow_from_directory(
    test_directory,
    target_size=(150, 150),
    batch_size=1,  # One image at a time
    class_mode='binary',  # Binary classification (taxiway vs runway)
    shuffle=False  # Keep order for predictions
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Make predictions on the test data
predictions = model.predict(test_generator)
predicted_classes = np.round(predictions).astype(int)

# Map predicted class indices to class labels
class_indices = test_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Invert the dictionary

true_classes = test_generator.classes

# plots confusion matrix (optional)
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), cm.shape[1]):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# computing confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plot_confusion_matrix(cm, list(class_labels.values()))

# outputs images with models predictions and labels
plt.figure(figsize=(12, 12))
for i in range(9):  # Display 9 sample predictions
    plt.subplot(3, 3, i + 1)
    img_path = test_generator.filepaths[i]
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    predicted_label = class_labels[predicted_classes[i][0]]
    true_label = class_labels[true_classes[i]]
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}")

plt.savefig('/directory/confusion_matrix.png')
print("Confusion matrix saved to /directory/confusion_matrix.png")

plt.show()
