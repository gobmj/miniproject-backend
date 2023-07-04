import cv2
import numpy as np
import os
import face_detect
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase
cred = credentials.Certificate('path/to/serviceAccountKey.json')  # Replace with the path to your service account key
firebase_admin.initialize_app(cred)

# Load training data from Firebase
db = firestore.client()
training_data_ref = db.collection('training_data')
training_data_docs = training_data_ref.get()

faces = []
labels = []
for doc in training_data_docs:
    data = doc.to_dict()
    faces.append(data['face'])
    labels.append(data['label'])

# Create and train the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

# Capture a snapshot using the camera
camera = cv2.VideoCapture(0)
return_value, test_img = camera.read()
cv2.imwrite("test-data/snapshot.jpg", test_img)
del(camera)

# Read the test image (snapshot taken with the camera)
test_img = "test-data/snapshot.jpg"
predicted_img, label_directories = face_detect.predict(test_img)

# Display the recognized label names
recognized_names = [get_label_name(label) for label in label_directories]
print("Recognized faces =", recognized_names)

# Store the recognized names in Firebase
recognized_students_ref = db.collection('recognized_students')
for name in recognized_names:
    recognized_students_ref.add({'name': name})
