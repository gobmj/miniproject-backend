import cv2
import numpy as np
import face_detect
import training_data

label = []
name = []
data = {1:"ds",2:"ar",3:"kp"}
def predict(test_img):
    img = cv2.imread(test_img).copy()
    print("\n")
    print("Face Prediction Running -\\-")
    face, rect, length = face_detect.face_detect(test_img)
    print(len(face), "faces detected.")
    for i in range(0, len(face)):
        labeltemp, confidence = face_recognizer.predict(face[i])
        label.append(labeltemp)
    return img, label

faces, labels = training_data.training_data("training-data")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

# Read the test image.
test_img = "test-data/test.jpg"
predicted_img, label = predict(test_img)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
for i in range(label.__sizeof__()):
    if label[i]==1:
        name.append("DQ")
    elif label[i]==2:
        name.append("AR")
    elif label[i]==3:
        name.append("KP")
    else:
        name.append("Unknown Face")
print("Recognized faces = ",name)

