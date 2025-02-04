import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3

# Load the trained model
model = load_model('model/traffic_sign_model.h5')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# List of traffic sign names (update with your classes)
traffic_signs = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
    'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to a PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Resize image to match model input shape
    img = img.resize((32, 32))  # Adjust size according to your model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    
    # Display the prediction
    sign_text = traffic_signs[predicted_class]
    cv2.putText(frame, sign_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Optionally, provide feedback via text-to-speech
    engine.say(sign_text)
    engine.runAndWait()
    
    # Display the resulting frame
    cv2.imshow('Traffic Sign Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()