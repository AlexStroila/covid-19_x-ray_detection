import os
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.metrics import f1_score
from image_generator import image_generator
from tqdm import tqdm

VALIDATION_DATA_FOLDER = r"Lung Segmentation Data\Val"

def calculate_f1_score(clf, val_gen, total_val_samples, batch_size):
    steps = total_val_samples // batch_size

    y_true = []
    y_pred = []

    for i in range(steps):
        (batch_x, batch_y), _ = next(val_gen)
        y_true.extend(batch_y)
        predictions = clf.predict(batch_x)
        y_pred.extend(predictions)

    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1

def train_classifier(gen, total_samples, batch_size):
    # Define and compile your model
    class_weights = [
        {0: 1.8, 1: 1, 2: 1.},
        {0: 2, 1: 1, 2: 1.},
        {0: 2.5, 1: 1, 2: 1.},
        {0: 3, 1: 1, 2: 1.},
        # Add more class weights here if you want to try them
    ]

    best_f1_score = 0
    best_clf = None
    best_weights = None

    # Define steps_per_epoch
    steps_per_epoch = total_samples // batch_size

    # Calculate the total number of images in the validation set
    total_val_samples = sum([len(files) for r, d, files in os.walk(VALIDATION_DATA_FOLDER)])

    # Instantiate a generator for validation data
    val_gen = image_generator(VALIDATION_DATA_FOLDER, batch_size)

    for weights in class_weights:
        clf = SGDClassifier(loss='log_loss', class_weight=weights, n_jobs=-1)

        # Loop over each batch from the generator
        for i in tqdm(range(steps_per_epoch), desc="Training progress"):
            (batch_x, batch_y), label_to_class = next(gen)
            clf.partial_fit(batch_x, batch_y, classes=np.unique(batch_y))

        # Test the classifier here and calculate the F1 score
        f1_score = calculate_f1_score(clf, val_gen, total_val_samples, batch_size)

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_clf = clf
            best_weights = weights

    print("Best weights: ", best_weights)
    print("Best F1 score: ", best_f1_score)

    return best_clf, label_to_class
