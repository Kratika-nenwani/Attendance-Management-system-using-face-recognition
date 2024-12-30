import cv2
import os
import numpy as np

def TrainImage():
    # Initialize the face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Directory for storing training images
    path = 'path/to/your/training/images'  # Change this to the correct path

    # Lists to store the images and labels
    faces = []
    ids = []

    # Iterate through the training images and labels
    for imagePath in os.listdir(path):
        img = cv2.imread(os.path.join(path, imagePath))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        id = int(imagePath.split('.')[0])  # Assuming the file name has the ID as part of the name

        faces.append(gray)
        ids.append(id)

    # Train the recognizer on the images
    recognizer.train(faces, np.array(ids))

    # Save the trained model
    recognizer.save("TrainingImageLabel/Trainner.yml")
    print("Training complete and model saved.")
