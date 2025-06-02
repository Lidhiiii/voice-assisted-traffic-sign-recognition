import numpy as np
import cv2
import os
import pickle
import pyttsx3
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import deque
import pandas as pd

# TTS Setup
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Dataset Preparation
DATASET_PATH = 'D:/LIDHI/PROJECT 24/myData'  # Path to your dataset
LABEL_FILE = 'D:/LIDHI/PROJECT 24/labels/labels.csv'  # Path to your label file

IMG_SIZE = 32
data = []
labels = []

# Load labels
label_data = pd.read_csv(LABEL_FILE)
CATEGORIES = label_data['Name'].tolist()

# Load and preprocess dataset
for category_id in range(len(CATEGORIES)):
    path = os.path.join(DATASET_PATH, str(category_id))
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized_array)
            labels.append(category_id)
        except Exception as e:
            pass

# Convert to numpy arrays and normalize
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
data = data / 255.0
labels = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, len(CATEGORIES))
y_test = to_categorical(y_test, len(CATEGORIES))

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Building the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(CATEGORIES), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('traffic_sign_model.h5')