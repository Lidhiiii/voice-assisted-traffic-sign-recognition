import numpy as np
import cv2
import pickle
import pyttsx3
import openai

# TTS Setup
engine = pyttsx3.init()

# OpenAI API Setup
openai.api_key = 'your-openai-api-key'

def speak(text):
    engine.say(text)
    engine.runAndWait()

def ask_llm(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=50
    )
    return response.choices[0].text.strip()

def estimate_distance(known_width, focal_length, per_width):
    # Distance Estimation Using Triangle Similarity
    return (known_width * focal_length) / per_width

#############################################

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75        # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# Traffic Sign Known Width (in meters)
KNOWN_WIDTH = 0.2
FOCAL_LENGTH = 500  # This needs to be calibrated for your camera

##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL
pickle_in = open("model_trained.p", "rb")  # rb = READ BYTE
model = pickle.load(pickle_in)

# Functions for image preprocessing
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

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
    if classNo >= 0 and classNo < len(classNames):
        return classNames[classNo]
    else:
        return "Unknown Class"


#def getClassName(classNo):
    #if classNo >= 0 and classNo < len(classNames):
        #return classNames[classNo]
    #else:
        #
        # return "Unknown Class"
#def getClassName(classNo):
    # Define the class names corresponding to the model output
    #classNames = ['Speed Limit 20 km/h', 'Speed Limit 30 km/h', ... ]  # Your full list here
    #return classNames[classNo]

while True:
    # READ IMAGE
    success, imgOriginal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)[0]
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        className = getClassName(classIndex)
        speak(f"Recognized sign: {className}")
        
        # Estimating the distance to the sign
        gray = grayscale(imgOriginal)
        edged = cv2.Canny(gray, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            distance = estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, w)
            speak(f"The sign is approximately {distance:.2f} meters away.")
        
        # Display the result
        cv2.putText(imgOriginal, f"CLASS: {className}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
