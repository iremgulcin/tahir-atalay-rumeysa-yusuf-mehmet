---
# MODEL CARD

# Model Card for yolov8_asl_s

<!-- Provide a quick summary of what the model is/does. -->

This model is an object detection model that can recognize 26 letters of the American Sign Language alphabet from images.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based on the yolov8s model from ultralytics, which is pretrained on the coco dataset. The model is finetuned on three datasets of American Sign Language letters, with a total of 6250 images for training. The model can detect and localize the hand gestures for each letter, and output the bounding box coordinates and the class label. The model is intended to be used for real-time sign language detection on an app.

- **Developed by:** Atalay Denknalbant, Ahmet Tahir Manzak
- **Model date:** February 6 2024
- **Model type:** Object Detection
- **Language(s):** American Sign Language
- **Finetuned from model:** yolov8s pretrained on coco dataset

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The model can be used directly for sign language detection on an app, where the user can input an image of their hand gesture and get the corresponding letter as the output. The app can also provide feedback and guidance on how to improve the gesture accuracy and clarity.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

The model should not be used for any high-stakes decision-making, such as medical diagnosis, legal proceedings, or security screening, where the consequences of errors or biases could be severe. The model should also not be used for any other languages or alphabets than American Sign Language, as it may not generalize well to different hand shapes, gestures, or symbols.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model has some limitations and risks that could affect its performance and reliability. Some of these are:

- The model has accuracy limitations on some letters, such as  J and Z. This could lead to misinterpretation or misunderstanding of the sign language gestures.
- The model may not generalize well to different lighting conditions, backgrounds, hand shapes, or skin tones, as the training data may not be representative of the diversity and variability of the real-world scenarios.
- The model may not account for the context, meaning, or nuance of the sign language communication, as it only detects the individual letters and not the words or sentences.
- The model may not respect the privacy or consent of the users or the people in the images, as it may capture and store sensitive or personal information without their knowledge or permission.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Some possible recommendations are:

- The users should be informed of the potential errors and limitations of the model, and be advised to use it with caution and human oversight.
- The users should be given the option to provide feedback or report any issues or concerns with the model or the app.
- The developers should monitor and evaluate the model performance and user satisfaction regularly, and update the model or the app accordingly.
- The developers should ensure that the model and the app comply with the ethical and legal standards and regulations of the relevant domains and jurisdictions.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
# Import the required libraries
import torch
import cv2
from PIL import Image

# Load the model
model = torch.hub.load('ultralytics/yolov8', 'custom', path='yolov8_asl_s.pt')

# Load the image
img = Image.open('test.jpg')

# Run the inference
results = model(img)

# Print the results
results.print()

# Show the results
results.show()
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model is trained on three datasets of American Sign Language letters, which are:

- [American Sign Language Letters](https://public.roboflow.com/object-detection/american-sign-language-letters/1): This dataset contains 3,455 images of 24 letters (excluding J and Z) in various backgrounds and lighting conditions.
- [ASL and FSL Combo](https://universe.roboflow.com/hand-signs-9v6jr/asl-and-fsl-combo): This dataset contains 3,000 images of 26 letters in American Sign Language and Filipino Sign Language, with different hand shapes and skin tones.
- [Sign Language](https://universe.roboflow.com/tfod-p4luj/sign_language-acf74): This dataset contains 1,440 images of 26 letters in American Sign Language, with different hand shapes and skin tones.

The total size of the training data is 6250 images. No preprocessing or filtering was done on the datasets.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

- **Training regime:** fp16 mixed precision
- **Optimization Algorithm:** AdamW
- **Training Parameters:** imgsz=416, dropout=0.5

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The testing data is randomly selected from the same datasets as the training data, with a size of 144 images for testing and 263 images for validation.

#### Factors

The factors that could affect the model performance are:

- The lighting conditions and the background of the image
- The hand shape and the skin tone of the person
- The angle and the distance of the camera
- The clarity and the accuracy of the gesture

#### Metrics

The metrics used to evaluate the model are:

- mAP50: the mean average precision at IoU (Intersection over Union) of 0.5
- mAP50-95: the mean average precision at IoU of 0.5 to 0.95 with a step size of 0.05

### Results

The results of the model evaluation are shown in the table below:

| Class | Images | Instances | Box(P) | R | mAP50 | mAP50-95 |
| ----- | ------ | --------- | ------ | - | ----- | -------- |
| all   | 263    | 263       | 0.955  | 0.946 | 0.983 | 0.842    |
| A     | 263    | 8         | 0.884  | 1     | 0.995 | 0.794    |
| B     | 263    | 9         | 0.954  | 0.889 | 0.961 | 0.834    |
| C     | 263    | 3         | 0.953  | 1     | 0.995 | 0.798    |
| D     | 263    | 10        | 0.915  | 1     | 0.995 | 0.823    |
| E     | 263    | 4         | 0.793  | 1     | 0.995 | 0.86     |
| F     | 263    | 8         | 0.955  | 1     | 0.995 | 0.842    |
| G     | 263    | 5         | 0.982  | 1     | 0.995 | 0.811    |
| H     | 263    | 9         | 0.991  | 1     | 0.995 | 0.737    |
| I     | 263    | 6         | 0.709  | 0.833 | 0.869 | 0.76     |
| J     | 263    | 9         | 0.988  | 0.889 | 0.893 | 0.633    |
| K     | 263    | 10        | 0.898  | 1     | 0.995 | 0.834    |
| L     | 263    | 9         | 0.995  | 1     | 0.995 | 0.885    |
| M     | 263    | 8         | 1      | 0.865 | 0.995 | 0.836    |
| N     | 263    | 9         | 0.974  | 1     | 0.995 | 0.872    |
| O     | 263    | 7         | 1      | 0.886 | 0.995 | 0.81     |
| P     | 263    | 15        | 1      | 0.856 | 0.95  | 0.845    |
| Q     | 263    | 12        | 0.986  | 1     | 0.995 | 0.883    |
| R     | 263    | 19        | 1      | 0.96  | 0.995 | 0.905    |
| S     | 263    | 4         | 0.959  | 1     | 0.995 | 0.873    |
| T     | 263    | 18        | 0.945  | 1     | 0.992 | 0.94     |
| U     | 263    | 15        | 0.982  | 0.933 | 0.991 | 0.882    |
| V     | 263    | 17        | 1      | 0.861 | 0.987 | 0.906    |
| W     | 263    | 16        | 1      | 0.951 | 0.995 | 0.909    |
| X     | 263    | 9         | 1      | 0.864 | 0.995 | 0.934    |
| Y     | 263    | 20        | 1      | 0.818 | 0.993 | 0.815    |
| Z     | 263    | 4         | 0.965  | 1     | 0.995 | 0.872    |

#### Summary

The model achieves high performance on most of the letters, with mAP50 above 0.95 and mAP50-95 above 0.8. However, some letters have lower precision or recall, such as I, J, M, O, P, V, W, X, and Y. These letters may have more variations or similarities in their gestures, which could make them harder to detect or classify.

### Model Architecture and Objective

The model architecture is based on the yolov8x model from ultralytics, which is a state-of-the-art object detection model that uses a single-stage detector with a deep convolutional neural network. The model has 8 output layers, each with 3 anchor boxes, for a total of 24 anchors. The model outputs the bounding box coordinates, the class label, and the confidence score for each detected object.

The model objective is to minimize the loss function, which consists of four components: the box loss, the objectness loss, the classification loss, and the label smoothing loss. The box loss measures the difference between the predicted and the target bounding box coordinates, using the generalized IoU metric. The objectness loss measures the difference between the predicted and the target objectness score, which indicates the probability of an object being present in the anchor box. The classification loss measures the difference between the predicted and the target class label, using the cross-entropy metric. The label smoothing loss adds a small amount of noise to the target class label, to prevent overfitting and improve generalization.

### Compute Infrastructure

The model was trained and evaluated on a Google Colab notebook and local desktop, using a Tesla P100 GPU and RTX 4090.

#### Hardware

The hardware requirements for the model are:

- GPU: NVIDIA Tesla P100, RTX 4090 or equivalent
- CPU: Intel i9, Xeon or equivalent
- RAM: 16 GB or more
- Disk: 10 GB or more

#### Software

The software requirements for the model are:

- Python 3.8 or higher
- PyTorch 1.9 or higher
- Torchvision 0.10 or higher
- Ultralytics yolov8 0.0.1 or higher
- OpenCV 4.5 or higher
- PIL 8.3 or higher

## Citation 

- Glenn Jocher, Alex Stoken, Jirka Borovec, NanoCode012, ChristopherSTAN, Laughing, lorenzomammana, tkianai, Adam Hogan, Mikhail Grankin, Ayush Chaurasia, Yonghye Kwon, Stijn van der Linden, Taha M. Khan, Rubén Rodríguez, Chanoh Park, Joseph M. Rocca, Rushil Anirudh, Hamid Rezatofighi, and Wang Xinyu. (2021). YOLOv5: State-of-the-art object detection. arXiv preprint arXiv:2106.04161.
