import cv2
from fer import FER
import tensorflow as tf

face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# using pre-trained FER model
emotion_detector = FER()

while True:
    ret, input_image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = input_image[y:y+h, x:x+w]
        emotions = emotion_detector.detect_emotions(face_region)

        if emotions:
            bounding_box = emotions[0]["box"]
            detected_emotions = emotions[0]["emotions"]
            cv2.rectangle(input_image, (x, y), (x + w, y + h), (100, 155, 255), 2)

            max_score = 0
            max_emotion = ""
            for emotion, score in detected_emotions.items():
                if score > max_score:
                    max_score = score
                    max_emotion = emotion

            for index, (emotion_name, score) in enumerate(detected_emotions.items()):
                color = (0, 0, 255) if emotion_name == max_emotion else (255, 0, 0)
                emotion_score = "{}: {:.2f}".format(emotion_name, score)
                cv2.putText(input_image, emotion_score, (x, y + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', input_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
