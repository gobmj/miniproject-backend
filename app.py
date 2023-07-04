from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_detect
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import urllib.request

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0)
camera.set(3, 800)  # Set width
camera.set(4, 600)  # Set height

# Initialize Firebase
cred = credentials.Certificate('mini-project-5efef-firebase-adminsdk-bdapf-2c93bf270f.json')  # Replace with the path to your service account key
firebase_admin.initialize_app(cred)

# Load training data from Firebase
db = firestore.client()
training_data_ref = db.collection('Students')
training_data_docs = training_data_ref.get()

faces = []
labels = []
for doc in training_data_docs:
    data = doc.to_dict()
    image_url = data['pictureURL']
    try:
        req = urllib.request.urlopen(image_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces.append(gray_image)
        labels.append(data['firstName'])
    except Exception as e:
        print('Failed to load image from URL:', image_url)

# Convert labels to a unique integer identifier for each name
label_ids = list(set(labels))
label_map = {label: i for i, label in enumerate(label_ids)}
labels_encoded = [label_map[label] for label in labels]

# Create and train the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels_encoded))


@app.route('/api/detect-face', methods=['POST'])
def detect_face():
    class_id = request.json['classId']
    print('Class ID:', class_id)

    # Capture image from the camera
    ret, frame = camera.read()
    if not ret:
        response = {'message': 'Failed to capture image'}
        return jsonify(response), 500

    # Convert image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection and recognition
    faces, label_directories = face_detect.predict(gray_image)

    # Display the recognized label names
    recognized_names = [get_label_name(label) for label in label_directories]
    print("Recognized faces =", recognized_names)

    # Store the recognized names in Firebase
    recognized_students_ref = db.collection('recognized_students')
    for name in recognized_names:
        recognized_students_ref.add({'name': name})

    #camera.release()

    # Return a response to the React application
    response = {'message': 'Successful'}
    return jsonify(response), 200


def get_label_name(label):
    # Retrieve the label name based on the label value from the Students collection
    student_docs = training_data_ref.where('firstName', '==', label_ids[label]).get()
    for doc in student_docs:
        data = doc.to_dict()
        return data['firstName']
    return 'Unknown'  # Return 'Unknown' if no matching label is found


if __name__ == '__main__':
    app.run()
