import numpy as np
import cv2
import pickle
import pyttsx3
import time
from collections import deque

# TTS Setup
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def estimate_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
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

# IMPORT THE TRAINED MODEL
with open("model_trained.p", "rb") as f:
    model = pickle.load(f)

# Functions for image preprocessing
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    return img.reshape(1, 32, 32, 1)

classNames = [
    'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
    'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
    'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
    'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
    'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
    'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons'
]

def getClassName(classNo):
    if 0 <= classNo < len(classNames):
        return classNames[classNo]
    return "Unknown Class"

frame_count = 0
process_every_n_frames = 3  # Process every 3rd frame

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
                        distance = estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, w)
                        speak(f"The sign is approximately {distance:.2f} meters away.")

                    cv2.putText(imgOriginal, f"CLASS: {className}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
