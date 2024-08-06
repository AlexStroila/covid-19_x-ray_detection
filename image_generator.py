import os
import glob
from PIL import Image
import numpy as np
from sklearn import preprocessing

# Your directory
TRAIN_DATA_FOLDER = r"Lung Segmentation Data\Train"

allowed_extensions = [".jpeg", ".png", ".jpg"]

def image_generator(input_dir, batch_size):
    """
    A generator that yields batches of images and labels.
    """
    image_files = []
    labels = []

    for class_folder_name in os.listdir(input_dir):
        class_folder_path = os.path.join(input_dir, class_folder_name)
        for image_path in glob.glob(os.path.join(class_folder_path, "*")):
            if os.path.splitext(image_path)[1] in allowed_extensions:
                image_files.append(image_path)
                labels.append(class_folder_name)

    # Perform encoding on labels
    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Create a mapping of encoded labels to class names
    label_to_class = dict(zip(le.transform(le.classes_), le.classes_))

    while True:
        # Shuffle the indices of the images
        indices = np.arange(len(image_files))
        np.random.shuffle(indices)

        for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]

            batch_input = []
            batch_output = []

            for index in excerpt:
                input = Image.open(image_files[index]).convert('L')
                if input.size != (256, 256):
                    input = input.resize((256, 256))
                # Normalize the pixel values (scale them between 0 and 1)
                input = np.array(input.getdata()) / 255.0
                output = encoded_labels[index]

                batch_input.append(input)
                batch_output.append(output)

            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield (batch_x, batch_y), label_to_class
