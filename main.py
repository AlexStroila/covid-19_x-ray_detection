import os
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from image_generator import image_generator
from train import train_classifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             accuracy_score, precision_recall_curve, 
                             average_precision_score, classification_report)
from sklearn.preprocessing import label_binarize

# Your directory
TRAIN_DATA_FOLDER = r"Lung Segmentation Data\Train"
TEST_DATA_FOLDER = r"Lung Segmentation Data\Test"

# Calculate the total number of images
total_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DATA_FOLDER)])

# Instantiate a generator
batch_size = 64
gen = image_generator(TRAIN_DATA_FOLDER, batch_size)

# Train the classifier
clf, label_to_class = train_classifier(gen, total_samples, batch_size)

# Instantiate a generator for test data
test_gen = image_generator(TEST_DATA_FOLDER, batch_size)

# Calculate the total number of images in the test set
total_test_samples = sum([len(files) for r, d, files in os.walk(TEST_DATA_FOLDER)])

# Define steps for the test set
test_steps = total_test_samples // batch_size
 
# Initialize an empty array for the true and predicted labels
y_true = []
y_pred = []

# Loop over each batch from the test generator
for i in tqdm(range(test_steps), desc="Evaluating test set"):
    (batch_x, batch_y), _ = next(test_gen)
    y_true.extend(batch_y)
    predictions = clf.predict(batch_x)
    y_pred.extend(predictions)

# Calculate the accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy on the test set: ", accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(len(label_to_class)))
class_names = [label_to_class[i] for i in range(len(label_to_class))]

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Binarize the output
y_true_bin = label_binarize(y_true, classes=range(len(label_to_class)))
y_pred_bin = label_binarize(y_pred, classes=range(len(label_to_class)))

n_classes = y_true_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Find the index of the 'COVID-19' class
covid_class_idx = None
for i, class_name in label_to_class.items():
    if class_name == 'COVID-19 3505':
        covid_class_idx = i
        break

# Plot ROC curves for each class
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.figure()  # Start a new figure for this plot
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(label_to_class[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class {0}'.format(label_to_class[i]))
    plt.legend(loc="lower right")
    plt.show()

# Compute Precision-Recall curve and average precision for each class
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
    average_precision = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])

    # Plot Precision-Recall curve for each class
    plt.figure()
    plt.step(recall, precision, where='post', label='Average precision-recall score: {0:0.2f}'.format(average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve of class {0}'.format(label_to_class[i]))
    plt.legend(loc="lower right")
    plt.show()

# Error analysis
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
