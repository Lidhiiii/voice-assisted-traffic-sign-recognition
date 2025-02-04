import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pyttsx3

# Step 1: Load and preprocess the dataset
def load_data(data_dir):
    classes = os.listdir(data_dir)
    images = []
    labels = []
    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path).resize((32, 32))
            images.append(np.array(image))
            labels.append(class_id)
    return np.array(images), np.array(labels)

# Step 2: Build the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(43, activation='softmax')  # Assuming 43 classes of traffic signs
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train the model
def train_model(model, train_images, train_labels):
    datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow(train_images, train_labels, subset='training')
    validation_generator = datagen.flow(train_images, train_labels, subset='validation')
    model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Step 4: Recognize traffic signs
def recognize_sign(model, image_path):
    img = Image.open(image_path).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Step 5: Text-to-Speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Main function to run the project
def main():
    data_dir = "D:\TRaffic sign recognition\Meta" # Replace with your dataset path
    train_images, train_labels = load_data(data_dir)
    model = create_model()
    train_model(model, train_images, train_labels)

    test_image_path = 'path_to_test_image.jpg'  # Replace with your test image path
    predicted_class = recognize_sign(model, test_image_path)
    traffic_signs = ['Sign1', 'Sign2', ..., 'Sign43']  # List of traffic sign names
    recognized_sign = traffic_signs[predicted_class]
    
    speak_text(f'The recognized traffic sign is {recognized_sign}')

if __name__ == "__main__":
    main()
