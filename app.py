# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# import face_detect
# from flask_cors import CORS
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import firestore, auth
# import urllib.request
# import face_detect

# app = Flask(__name__)
# CORS(app)

# camera = cv2.VideoCapture(0)
# camera.set(3, 800)  # Set width
# camera.set(4, 600)  # Set height

# # Initialize Firebase
# cred = credentials.Certificate('mini-project2-1a7d2-firebase-adminsdk-1b242-f44442856b.json')  # Replace with the path to your service account key
# firebase_admin.initialize_app(cred)

# # Load training data from Firebase
# db = firestore.client()
# training_data_ref = db.collection('Students')
# training_data_docs = training_data_ref.get()

# recognized_students_ids = []

# faces = []
# labels = []
# for doc in training_data_docs:
#     data = doc.to_dict()
#     image_url = data['pictureURL']
#     try:
#         req = urllib.request.urlopen(image_url)
#         arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#         image = cv2.imdecode(arr, -1)
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#         faces.append(gray_image)
#         labels.append(data['firstName'])
#     except Exception as e:
#         print('Failed to load image from URL:', image_url)

# # Convert labels to a unique integer identifier for each name
# label_ids = list(set(labels))
# label_map = {label: i for i, label in enumerate(label_ids)}
# labels_encoded = [label_map[label] for label in labels]

# # Create and train the face recognizer
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.train(faces, np.array(labels_encoded))


# @app.route('/api/detect-face', methods=['POST'])
# def detect_face():
#     class_id = request.json['classId']
#     print('Class ID:', class_id)

#     # Capture image from the camera
#     ret, frame = camera.read()
#     if not ret:
#         response = {'message': 'Failed to capture image'}
#         return jsonify(response), 500

#     recognized_students_ids = []

#     # Convert image to BGR format (if not already)
#     if len(frame.shape) == 2:
#         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#     # Convert image to grayscale
#     gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Perform face detection and recognition
#     recognized_faces, label_directories = face_detect.predict(gray_image, face_recognizer, label_ids)

#     # Check if faces are detected
#     if not recognized_faces:
#         response = {'message': 'No faces detected'}
#         return jsonify(response), 404

#     # Display the recognized label names
#     recognized_names = [get_label_name(label, class_id) if label is not None else 'Unknown' for label in label_directories]
#     print("Recognized faces =", recognized_names)

#     # Store the recognized names in Firebase with document IDs
#     recognized_students_ref = db.collection('recognized_students')
#     for name in recognized_names:
#         student_docs = training_data_ref.where('firstName', '==', name).where('classId', '==', class_id).get()
#         for doc in student_docs:
#             recognized_students_ref.add({'name': name})
#             recognized_students_ids.append(doc.id)

#     # Return the recognized student document IDs as JSON
#     return jsonify(student_ids=recognized_students_ids), 200

# def get_label_name(label, class_id):
#     # Retrieve the label name based on the label value from the Students collection
#     student_docs = training_data_ref.where('firstName', '==', label_ids[label]).where('classId', '==', class_id).get()
#     for doc in student_docs:
#         data = doc.to_dict()
#         return data['firstName']
#     return 'Unknown'  # Return 'Unknown' if no matching label is found

# @app.route('/api/delete-user', methods=['POST'])
# def delete_user():
#     user_id = request.json['userId']
#     try:
#         auth.delete_user(user_id)
#         response = {'message': 'User deleted successfully'}
#         return jsonify(response), 200
#     except firebase_admin.auth.FirebaseAuthError as e:
#         response = {'error': str(e)}
#         return jsonify(response), 500


# if __name__ == '__main__':
#     app.run()




from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_detect
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, auth
import urllib.request
import face_detect

app = Flask(__name__)
CORS(app)

# Initialize Firebase
cred = credentials.Certificate('mini-project2-1a7d2-firebase-adminsdk-1b242-f44442856b.json')  # Replace with the path to your service account key
firebase_admin.initialize_app(cred)

# Load training data from Firebase
db = firestore.client()
training_data_ref = db.collection('Students')
training_data_docs = training_data_ref.get()

recognized_students_ids = []

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

    # Provide the path of the image in your local machine for testing
    image_path = r'C:\Users\govin\Downloads\pics(0)\test.png'

    # Read the image from the local machine
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection and recognition
    recognized_faces, label_directories = face_detect.predict(gray_image, face_recognizer, label_ids)
    # Check if faces are detected
    if not recognized_faces:
        response = {'message': 'No faces detected'}
        return jsonify(response), 404

    # Display the recognized label names
    recognized_names = [get_label_name(label, class_id) if label is not None else 'Unknown' for label in label_directories]
    print("Recognized faces =", recognized_names)

    # Store the recognized names in Firebase with document IDs
    recognized_students_ref = db.collection('recognized_students')
    for name in recognized_names:
        student_docs = training_data_ref.where('firstName', '==', name).where('classId', '==', class_id).get()
        for doc in student_docs:
            recognized_students_ref.add({'name': name})
            recognized_students_ids.append(doc.id)

    # Return the recognized student document IDs as JSON
    return jsonify(student_ids=recognized_students_ids), 200

def get_label_name(label, class_id):
    # Retrieve the label name based on the label value from the Students collection
    student_docs = training_data_ref.where('firstName', '==', label_ids[label]).where('classId', '==', class_id).get()
    for doc in student_docs:
        data = doc.to_dict()
        return data['firstName']
    return 'Unknown'  # Return 'Unknown' if no matching label is found

@app.route('/api/delete-user', methods=['POST'])
def delete_user():
    user_id = request.json['userId']
    try:
        auth.delete_user(user_id)
        response = {'message': 'User deleted successfully'}
        return jsonify(response), 200
    except firebase_admin.auth.FirebaseAuthError as e:
        response = {'error': str(e)}
        return jsonify(response), 500


if __name__ == '__main__':
    app.run()

