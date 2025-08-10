# Project Report: Maize Disease Classification Using Transfer Learning and Custom CNN

## 1. Introduction
The project aims to classify two maize diseases, **Army Worm** and **Cutworm**, based on image data. This is crucial for early detection and mitigation of crop damage caused by these diseases. Two approaches were implemented:

1. **Transfer Learning**: Using a pre-trained ResNet50 model.
2. **Custom Convolutional Neural Network (CNN)**: A model designed from scratch.

The goal was to compare the accuracy, efficiency, and generalization ability of these approaches.

---

## 2. Methodology

### 2.1 Data Preparation
- **Dataset**:
  - Total Images: 200 (160 for training and 40 for validation).
  - Two classes: Army Worm (100 images) and Cutworm (100 images).

- **Data Augmentation**:
  - Techniques applied: Rescaling, Rotation, Zooming, Flipping, and Shifting.
  - Purpose: To prevent overfitting and increase generalization.

### 2.2 Model Architectures
1. **Transfer Learning**:
   - **Base Model**: ResNet50 (pre-trained on ImageNet).
   - **Custom Layers**: Added Dense and Dropout layers for classification.
   - **Parameters**: Frozen the base layers, only fine-tuned the top layers.

2. **Custom CNN**:
   - **Architecture**:
     - 3 Convolutional Layers with ReLU activation and MaxPooling.
     - Fully Connected Dense layers.
     - Dropout for regularization.
   - Designed for a lightweight implementation.

### 2.3 Training Setup
- **Optimizer**: Adam.
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Accuracy.
- **Batch Size**: 16.
- **Epochs**: 25.

### 2.4 Tools and Libraries
- TensorFlow, Keras, Matplotlib, NumPy, Pandas, scikit-learn.

---

## 3. Results

### 3.1 Training Performance
| **Model**              | **Training Accuracy** | **Validation Accuracy** | **Training Time** |
|-------------------------|-----------------------|--------------------------|--------------------|
| Transfer Learning (ResNet50) | 54.53%                | 60%                      | ~59 minutes        |
| Custom CNN              | 86.53%                | 77.50%                   | ~6 minutes         |

### 3.2 Observations
1. **Transfer Learning**:
   - Achieved moderate accuracy with some limitations in generalization.
   - Training time was significantly longer compared to the Custom CNN.

2. **Custom CNN**:
   - Achieved higher accuracy than ResNet50 within fewer epochs.
   - Faster training and demonstrated better efficiency for lightweight implementation.

### 3.3 Overfitting/Underfitting
- Both models showed slight overfitting, mitigated by augmentation and Dropout.

### 3.4 Prediction Visualization
- The evaluation script now includes functionality to save validation images with annotated predicted labels in a specified output directory. This allows for visual inspection of model predictions, facilitating qualitative analysis of classification performance.

---

## 4. Folder Structure
The project directory is organized as follows:

```
Maize_Disease_Classification/
│
├── Dataset/
│   ├── train/
│   │   ├── Army_Worm/
│   │   └── Cutworm/
│   └── validation/
│       ├── Army_Worm/
│       └── Cutworm/
├── models/
│   └── trained_model.h5
├── outputs/  # New: Directory for saving predicted images (created during evaluation)
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt
```

- **Dataset/**: Contains training and validation images, split into subdirectories for each class (`Army_Worm` and `Cutworm`).
- **models/**: Stores the trained model file (`trained_model.h5`).
- **outputs/**: Stores annotated validation images with predicted labels after evaluation.
- **scripts/**: Contains Python scripts for training (`train.py`) and evaluation (`evaluate.py`).
- **requirements.txt**: Lists required Python libraries.
- **README.md**: Project documentation.

---

## 5. How to Run the Code

### Prerequisites
- Ensure Python 3.8+ is installed.
- Install required libraries by running:
  ```
  pip install -r requirements.txt
  ```

### Training the Model
To train the model, use the following command:

```bash
python scripts/train.py --train-dir Dataset/train --validation-dir Dataset/validation --model-type resnet50 --epochs 10 --model-save-path ./models/trained_model.h5 --learning-rate 0.0001 --batch-size 32 --target-size 256 256 --verbose 1
```

- **Arguments**:
  - `--train-dir`: Path to the training dataset directory.
  - `--validation-dir`: Path to the validation dataset directory.
  - `--model-type`: Specify the model (`resnet50` or `custom_cnn`).
  - `--epochs`: Number of training epochs.
  - `--model-save-path`: Path to save the trained model.
  - `--learning-rate`: Learning rate for the optimizer.
  - `--batch-size`: Batch size for training.
  - `--target-size`: Image dimensions (width, height).
  - `--verbose`: Verbosity level (0 or 1).

### Evaluating the Model
To evaluate the trained model and save predictions, use the following command:

```bash
python scripts/evaluate.py --validation-dir Dataset/validation --model-path ./models/trained_model.h5 --output-dir ./outputs --batch-size 32 --target-size 256 256
```

- **Arguments**:
  - `--validation-dir`: Path to the validation dataset directory.
  - `--model-path`: Path to the trained model file.
  - `--output-dir`: Path to the directory where annotated prediction images will be saved.
  - `--batch-size`: Batch size for evaluation.
  - `--target-size`: Image dimensions (width, height).

The evaluation process computes validation loss and accuracy, then saves each validation image with the predicted label annotated and incorporated into the filename for easy review.

---

## 6. Conclusion

### 6.1 Findings
- Both models demonstrated similar performance on this dataset.
- Transfer Learning with ResNet50 showed moderate results, but training time was longer.
- Custom CNN achieved higher accuracy and faster training, indicating its suitability for this task.

### 6.2 Future Work
- Expand the dataset to include more images and other maize diseases.
- Experiment with state-of-the-art models like EfficientNet or Vision Transformers.
- Utilize advanced augmentation techniques like CutMix or MixUp.
- Incorporate model ensemble techniques to boost accuracy.

### 6.3 Applications
- Early disease detection in maize cultivation.
- Deployment on edge devices for real-time disease monitoring.
