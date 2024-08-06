import Augmentor
import os

# Your directory
TRAIN_DATA_FOLDER = r"Lung Segmentation Data\Train"

allowed_extensions = [".jpeg", ".png", ".jpg"]

# List of augmentations
augmentations = ["rotate", "translate", "rescale", "brightness_contrast"]

def augment_images(input_dir, output_dir, augment_type):
    """
    This function takes an input directory of images, applies specified augmentation, and saves the images in the output directory.
    """

    # Initialize the pipeline (don't load standard augmentation operations)
    p = Augmentor.Pipeline(input_dir, output_dir, save_format="JPEG")

    if augment_type == "rotate":
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    elif augment_type == "translate":
        p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=2)
    elif augment_type == "rescale":
        p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    elif augment_type == "brightness_contrast":
        p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)
        p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)
    
    p.sample(len(p.augmentor_images))

# Process all classes
for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = os.path.join(TRAIN_DATA_FOLDER, class_folder_name)
    
    for augment in augmentations:
        output_dir = os.path.join(TRAIN_DATA_FOLDER, f"{class_folder_name}_{augment}")

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Augment the images
        augment_images(class_folder_path, output_dir, augment)
