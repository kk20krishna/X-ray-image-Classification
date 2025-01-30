# Chest X-Ray Image Classification using Transfer Learning

## Introduction

This project focuses on developing an image classification model to detect pleural effusion in chest X-ray images using transfer learning. Pleural effusion is a condition characterized by the abnormal accumulation of fluid in the pleural space, the area between the lungs and the chest wall. Early and accurate detection of pleural effusion is crucial for timely medical intervention and improved patient outcomes. This project aims to automate the detection process, reducing the reliance on manual interpretation and potential inter-observer variability.

## Dataset

The project utilizes the CXR dataset, a publicly available collection of chest X-ray images labeled with the presence or absence of pleural effusion. The dataset comprises 1107 images, categorized into 'effusion' and 'nofinding' classes. This dataset was downloaded using the `kagglehub` library directly within the Google Colab environment.

Dataset - https://www.kaggle.com/datasets/kk20krishna/cxr-data

## Methodology

### Data Preprocessing

- **Data Augmentation:** We applied data augmentation techniques to the 'effusion' class images using the `Augmentor` library to increase the dataset size and improve model robustness. This included random rotations, and potentially other augmentations like shear, brightness, and contrast adjustments.
- **Dataset Splitting:** The dataset was split into training and validation sets using `keras.utils.image_dataset_from_directory` with a validation split of 0.2. This ensured that the model was evaluated on unseen data to assess its generalization performance.

### Model Development

- **Transfer Learning:** We employed transfer learning by utilizing the ResNet50V2 model pre-trained on ImageNet as the base model. This approach leverages the knowledge gained from a large and diverse dataset to accelerate the training process and enhance model performance on the target task.
- **Model Architecture:** The top classification layer of the ResNet50V2 model was replaced with new layers suitable for binary classification. We used Global Average Pooling to reduce parameters and added Dense layers with Dropout for regularization.
- **Model Training:** The model was trained for 50 epochs using the training dataset and evaluated on the validation dataset. We used the 'adam' optimizer and 'categorical_crossentropy' loss function, along with metrics such as accuracy, AUC, precision, and recall to monitor training progress.

## Results

**The trained model achieved AUC: 0.9521, Accuracy: 0.9080, Precision: 0.9080 and Recall: 0.9080 on the validation dataset,** indicating the model's performance in correctly classifying images into effusion and nofinding categories. Visualization of predictions on sample images further demonstrated the model's ability to accurately identify pleural effusion.

## Conclusion

This project demonstrates the successful application of transfer learning for pleural effusion detection in chest X-ray images. The model shows promising results and can potentially aid in assisting healthcare professionals with diagnosis. Further improvements could involve exploring different model architectures, fine-tuning hyperparameters, and incorporating a larger and more diverse dataset.

## Acknowledgements

We acknowledge the creators of the CXR dataset and the developers of the libraries used in this project.
