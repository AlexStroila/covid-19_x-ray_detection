# COVID-19 Detection Using Chest X-Ray Images

## Abstract
This project introduces a novel approach employing multinomial logistic regression combined with Stochastic Gradient Descent (SGD) to classify chest X-ray images into COVID-19, Non-COVID, and Normal categories. By leveraging data augmentation techniques, the model achieved a respectable accuracy of 72%, suggesting potential utility in medical diagnostics.

## Introduction
The study addresses the critical need for rapid and accurate COVID-19 diagnostics using simple yet effective machine learning models, highlighting the importance of chest X-rays (CXR) in detecting lung pathologies related to COVID-19.

## Problem Framework
The research focuses on a pixel-by-pixel examination of CXR images using multinomial logistic regression, an extension of logistic regression designed for multi-class problems, offering a simple and interpretable model suitable for medical settings where resources may be limited.

## Data & Methodology
Data sourced from a publicly accessible Kaggle dataset was enhanced with data augmentation techniques, significantly increasing the dataset size and diversity, which helped improve model robustness and performance.

### Key Techniques Used
- **Multinomial Logistic Regression**: For classifying images into multiple categories.
- **Stochastic Gradient Descent (SGD)**: To efficiently handle large-scale datasets.
- **Data Augmentation**: To increase dataset size and variability, including rotations and brightness adjustments.

## Results & Discussions
The model demonstrated promising results with an F1 score of 72% on the augmented dataset. Data augmentation played a crucial role in enhancing the model's predictive accuracy and robustness.

### Confusion Matrix for the Original Dataset
![Confusion Matrix for the Original Dataset](/images/Confusion_Matrix_Original.png)
*Confusion matrix for the original dataset showing the model's performance across different classes.*

### Confusion Matrix for the Augmented Dataset
![Confusion Matrix for the Augmented Dataset](/images/Confusion_Matrix_Augmented.png)
*Confusion matrix for the augmented dataset illustrating improved performance and generalization.*


## Conclusions
The findings underscore the viability of using simpler models, like multinomial logistic regression, for complex image classification tasks in medical imaging, particularly when computational resources are limited.

## How to Use This Repository
- **Download the Dataset**: The original dataset can be accessed from [Kaggle COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). However, in the repository you will find the original plus the augmented images directly.
- **Run the Model**: Simply execute `main.py` to run the model. For details on the model parameters and setup, refer to `train.py`. 
- **Explore Further**: The methodology and full results are documented in the accompanying [PDF report](COVID-19_Detection_Chest_XRay_Multinomial_Logistic_Report.pdf).
