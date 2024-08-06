# COVID-19 Detection Using Chest X-Ray Images

## Abstract
This project introduces a novel approach employing multinomial logistic regression combined with Stochastic Gradient Descent (SGD) to classify chest X-ray images into COVID-19, Non-COVID, and Normal categories. By leveraging data augmentation techniques, the model achieved a respectable accuracy of 72%, suggesting potential utility in medical diagnostics.

## Introduction
The study addresses the critical need for rapid and accurate COVID-19 diagnostics using simple yet effective machine learning models, highlighting the importance of chest X-rays (CXR) in detecting lung pathologies related to COVID-19.

## Problem Framework
The research focuses on a **pixel-by-pixel** examination of CXR images using multinomial logistic regression, an extension of logistic regression designed for multi-class problems, offering a simple and interpretable model suitable for medical settings where resources may be limited.

## Data & Methodology
Data sourced from a publicly accessible Kaggle dataset was enhanced with data augmentation techniques (i.e., rotation, translation, rescaling, and brightness adjustment), significantly increasing the dataset size and diversity, which helped improve model robustness and performance.

### Data Augmentation Examples

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: nowrap;">
  <div style="text-align: center;">
    <img src="/images/Original.png" alt="Original" style="width: 100px; display: block; margin: 0 auto;">
    <span><strong>Original</strong></span>
  </div>
  <div style="text-align: center;">
    <img src="/images/Translate.JPEG" alt="Translation" style="width: 100px; display: block; margin: 0 auto;">
    <span><strong>Translated</strong></span>
  </div>
  <div style="text-align: center;">
    <img src="/images/Rotate.JPEG" alt="Rotation" style="width: 100px; display: block; margin: 0 auto;">
    <span><strong>Rotated</strong></span>
  </div>
  <div style="text-align: center;">
    <img src="/images/Rescale.JPEG" alt="Rescale" style="width: 100px; display: block; margin: 0 auto;">
    <span><strong>Rescaled</strong></span>
  </div>
  <div style="text-align: center;">
    <img src="/images/Brightness.JPEG" alt="Brightness Adjustment" style="width: 100px; display: block; margin: 0 auto;">
    <span><strong>Brightness Adjusted</strong></span>
  </div>
</div>




### Key Techniques Used
- **Multinomial Logistic Regression**: For classifying images into multiple categories.
- **Stochastic Gradient Descent (SGD)**: To efficiently handle large-scale datasets.
- **Data Augmentation**: To increase dataset size and variability, including rotations and brightness adjustments.

## Results & Discussions
The model demonstrated promising results with an F1 score of 72% on the augmented dataset. Data augmentation played a crucial role in enhancing the model's predictive accuracy and robustness.

<p float="left">
  <img src="/images/Confusion_Matrix_Original.png" width="49%" />
  <img src="/images/Confusion_Matrix_Augmented.png" width="49%" /> 
</p>

**Confusion matrices: On the left is the original dataset, and on the right is the augmented dataset, illustrating improved performance and generalization.**

## Conclusions
The findings underscore the viability of using simpler models, like multinomial logistic regression, for complex image classification tasks in medical imaging, particularly when computational resources are limited.

## How to Use This Repository
- **Download the Dataset**: The original dataset can be accessed from [Kaggle COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). However, in the repository you will find the original plus the augmented images directly.
- **Run the Model**: Simply execute `main.py` to run the model. For details on the model parameters and setup, refer to `train.py`. 
- **Explore Further**: The methodology and full results are documented in the accompanying [PDF report](COVID-19_Detection_Chest_XRay_Multinomial_Logistic_Report.pdf).
