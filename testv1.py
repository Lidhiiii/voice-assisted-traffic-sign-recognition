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
from tensorflow.keras.models import load_model


# Testing Phase
#############################################

frameWidth = 320  # Lower resolution to reduce lag
frameHeight = 240
brightness = 150
threshold = 0.90  # Increased threshold for more confident predictions
font = cv2.FONT_HERSHEY_SIMPLEX

# Traffic Sign Known Width (in meters)
KNOWN_WIDTH = 0.2
FOCAL_LENGTH = 500  # This needs to be calibrated for your camera

# Cooldown and Prediction Smoothing Setup
cooldown_time = 5  # seconds
last_spoken_time = 0
recent_predictions = deque(maxlen=10)  # Keep the last 10 predictions

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS to reduce CPU usage

# Load the trained model
#model = tensorflow.keras.models.load_model('traffic_sign_model.h5')
model = load_model('traffic_sign_model.h5')
# Functions for image preprocessing
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    return img.reshape(1, 32, 32, 1)

def getClassName(classNo):
    if 0 <= classNo < len(CATEGORIES):
        return CATEGORIES[classNo]
    return "Unknown Class"

frame_count = 0
process_every_n_frames = 5  # Process every 5th frame for even less CPU usage

while True:
    success, imgOriginal = cap.read()
    if not success:
        continue

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        continue

    img = preprocessing(imgOriginal)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)[0]
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        recent_predictions.append(classIndex)

        if len(recent_predictions) == recent_predictions.maxlen:
            most_common_class = max(set(recent_predictions), key=recent_predictions.count)

            if most_common_class == classIndex:
                current_time = time.time()

                if current_time - last_spoken_time > cooldown_time:
                    className = getClassName(classIndex)
                    speak(f"Recognized sign: {className}")
                    last_spoken_time = current_time

                    gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
                    edged = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w
                        speak(f"The sign is approximately {distance:.2f} meters away.")

                    cv2.putText(imgOriginal, f"CLASS: {className}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()