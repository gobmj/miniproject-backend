import cv2

def predict(image):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract face regions and return them along with the corresponding label directories
    face_regions = []
    label_directories = []
    for (x, y, w, h) in faces:
        face_regions.append(image[y:y+h, x:x+w])
        label_directories.append(0)  # Replace with your logic to assign label directories

    return face_regions, label_directories
