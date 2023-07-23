import cv2

def to_rgb(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return rgb_image

def predict(image, face_recognizer, label_ids):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if faces are detected
    if len(faces) == 0:
        return [], []

    # Extract face regions and recognize them
    recognized_faces = []
    label_directories = []
    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]

        # Convert the face region to RGB before calling cv2.cvtColor()
        face_region = to_rgb(face_region)

        # Perform face recognition on the grayscale face region
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        label, confidence = face_recognizer.predict(gray_face)

        # Check if the confidence level is within a reasonable threshold
        if confidence < 100:
            label_directories.append(label)
            recognized_faces.append(face_region)
        else:
            label_directories.append(None)

    return recognized_faces, label_directories

