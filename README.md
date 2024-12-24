# skin-cancer-detection-and-segmentation
Skin cancer detection and segmentation using deep learning involves building a model that can analyze and classify skin lesions from images, as well as segment the lesion regions to isolate them for further analysis. For this, Convolutional Neural Networks (CNNs) are commonly used due to their effectiveness in image classification tasks. Below is a Python-based solution using TensorFlow/Keras to build a skin cancer detection and segmentation system.
Steps:

    Preprocessing the Data: Loading and normalizing skin lesion images.
    Building the CNN for Classification: Using a CNN architecture for skin cancer classification (Benign/Malignant).
    Building a Segmentation Model: Using a U-Net architecture for semantic segmentation to isolate the skin lesions.

Step 1: Install Required Libraries

pip install tensorflow numpy matplotlib opencv-python scikit-learn

Step 2: Data Preparation and Preprocessing

Assuming you have a dataset such as the ISIC (International Skin Imaging Collaboration) dataset that contains images of skin lesions with labels (e.g., benign, malignant) and corresponding segmentation masks.

import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define image size
IMAGE_SIZE = 224  # You can choose other sizes based on your dataset

def load_image(image_path, size=(IMAGE_SIZE, IMAGE_SIZE)):
    # Read image from path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, size)  # Resize image
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Load dataset images and their corresponding labels
def load_dataset(image_dir, mask_dir, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
    images = []
    masks = []
    
    for img_name in os.listdir(image_dir):
        if img_name.endswith(".jpg"):  # Assuming images are in .jpg format
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '_mask.png'))  # Assuming masks are in _mask.png format
            
            image = load_image(img_path, size=image_size)
            mask = load_image(mask_path, size=image_size)
            
            images.append(image)
            masks.append(mask)
    
    return np.array(images), np.array(masks)

# Example: Assuming you have 'images/' and 'masks/' directories
image_dir = 'images/'
mask_dir = 'masks/'
X, y = load_dataset(image_dir, mask_dir)

Step 3: Data Augmentation

Using ImageDataGenerator for augmenting the dataset, which is useful for improving the robustness of the model.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen.fit(X)  # Fit the generator to the images

Step 4: Model Architecture for Skin Cancer Detection (Classification)

We’ll use a simple CNN model for binary classification (Benign or Malignant).

from tensorflow.keras import layers, models

def create_classification_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # Binary classification (Benign/Malignant)
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Create and train the model
classification_model = create_classification_model()
classification_model.summary()

# Train on the data
classification_model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

Step 5: Model Architecture for Skin Cancer Segmentation (U-Net)

Now, we will create a U-Net model for segmenting the skin lesions.

from tensorflow.keras import layers, models

def create_unet_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(input_shape)
    
    # Encoding path (Contracting Path)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    b = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    b = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(b)
    
    # Decoding path (Expansive Path)
    u1 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(b)
    u1 = layers.concatenate([u1, c3])
    c4 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u1)
    c4 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c4)
    
    u2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u2)
    c5 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c5)
    
    u3 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u3)
    c6 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c6)
    
    # Output layer
    output = layers.Conv2D(1, (1, 1), activation="sigmoid")(c6)
    
    model = models.Model(inputs, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

# Create and train the segmentation model
unet_model = create_unet_model()
unet_model.summary()

# Train the model
unet_model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

Step 6: Predicting and Evaluating the Models

After training the models, we can use them to predict the classification and segmentation results for new images.

# Predict using classification model (Benign/Malignant)
classification_prediction = classification_model.predict(new_image)  # new_image is a preprocessed image
print("Classification Prediction:", classification_prediction)

# Predict using U-Net for segmentation
segmentation_prediction = unet_model.predict(new_image)  # new_image should be preprocessed
segmented_image = (segmentation_prediction[0] > 0.5).astype(np.uint8)  # Binary thresholding to get binary mask

Step 7: Visualize the Results

Finally, you can visualize the segmented image along with the original image:

import matplotlib.pyplot as plt

# Visualize original vs segmented
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(new_image)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap="gray")
plt.title("Segmented Image")
plt.show()

Conclusion:

This is a basic framework for skin cancer detection and segmentation using deep learning. The classification model detects whether the lesion is benign or malignant, while the segmentation model isolates the skin lesion. For real-world use, you would need a large, well-annotated dataset for training, as well as optimization and fine-tuning of the models.
