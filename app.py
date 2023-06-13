from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_detect
import training_data
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# label = []
# name = []
# data = {1: "ds", 2: "ar", 3: "kp"}

@app.route('/api/detect-face', methods=['POST'])
def detect_face():
    # Access the image sent from the React application
    image = request.files['image']
    class_id = request.form['classid']

    print('Image received:', image.filename)
    print('Class ID:', class_id)

    # Perform face detection and recognition

    # Use the provided code for face detection and recognition here

    # def predict(image):
    #     img = cv2.imread(image).copy()
    #     print("\n")
    #     print("Face Prediction Running -\\-")
    #     face, rect, length = face_detect.face_detect(test_img)
    #     print(len(face), "faces detected.")
    #     for i in range(0, len(face)):
    #         labeltemp, confidence = face_recognizer.predict(face[i])
    #         label.append(labeltemp)
    #     return img, label
    #
    # faces, labels = training_data.training_data("training-data")
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # face_recognizer.train(faces, np.array(labels))
    #
    # # Read the test image.
    # test_img = "test-data/test.jpg"
    # predicted_img, label = predict(test_img)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    #
    # for i in range(label.__sizeof__()):
    #     if label[i] == 1:
    #         name.append("DQ")
    #     elif label[i] == 2:
    #         name.append("AR")
    #     elif label[i] == 3:
    #         name.append("KP")
    #     else:
    #         name.append("Unknown Face")
    # print("Recognized faces = ", name)

    # Store the detected faces or relevant information in Firebase
    # Use the Firebase SDK to store the data as needed

    # Return a response to the React application
    response = {'message': "Successful"}
    return jsonify(response), 200
