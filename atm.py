import tensorflow as tf
print(f"TensorFlow version in face_detection_yolo.py: {tf.__version__}")

import cv2
import time
from ultralytics import YOLO
from fer import FER

# YOLO 모델 관련 변수 설정
class_names = {0: 'ear', 1: 'eye', 2: 'mouth', 3: 'nose'}
model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

# pndetection 관련 변수 설정
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
emotion_detector = FER()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0

# main2 코드
def main2():
    global count
    start_time = time.time()
    face_confirmed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=0.6)
        detected_classes = [class_names.get(int(box.cls), 'unknown') for box in results[0].boxes]
        detected_classes_set = set(detected_classes)
        required_classes = {'eye', 'nose', 'mouth'}
        confirmed = required_classes.issubset(detected_classes_set)

        if confirmed:
            print("Face Confirmed")
            cv2.putText(frame, 'Confirmed', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('YOLOv8 Webcam', frame)
            cv2.waitKey(2000)
            face_confirmed = True
            break
        else:
            if time.time() - start_time >= 2:
                count += 1
                start_time = time.time()
                print(f"Face not detected for {count} times")
                cv2.putText(frame, 'Not Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                if count > 2:
                    print("Face not detected")
                    break

        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8 Webcam', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if face_confirmed:
        return True
    return False

# pndetection 코드
def pndetection():
    global count
    print("pndetection function started")
    start_time = time.time()
    extract_duration = 3
    negative_instance_count = 0
    positive_instance_count = 0

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

                if max_emotion in ["angry", "disgust", "fear", "sad", "surprise"]:
                    overall_emotion = "negative"
                    negative_instance_count += 1
                else:
                    overall_emotion = "positive"
                    positive_instance_count += 1

                cv2.putText(input_image, f"Overall: {overall_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', input_image)

        if time.time() - start_time > extract_duration:
            if negative_instance_count > positive_instance_count:
                count += 1
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Total Count : {count}")

# main2 실행
if main2():
    print("main2 completed successfully, starting pndetection")
    # main2가 성공적으로 완료되면 pndetection 실행
    pndetection()
else:
    print("main2 did not complete successfully, skipping pndetection")

cap.release()
cv2.destroyAllWindows()