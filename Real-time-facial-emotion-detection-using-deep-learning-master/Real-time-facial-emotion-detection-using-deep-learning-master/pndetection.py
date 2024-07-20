import cv2
from fer import FER
import tensorflow as tf
import time

face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# using pre-trained FER model
emotion_detector = FER()

#count 0으로 초기화
count = 0

#3초 동안 감정 추출을 위한 초기화
start_time = time.time()
extract_duration = 10

while True:
    ret, input_image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    negative_instance_count = 0
    positive_instance_count = 0

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

            # 감정 결과 표시
            if max_emotion in ["angry", "disgust", "fear", "sad", "surprise"]:
                overall_emotion = "negative"
                negative_instance_count += 1
    
            else:
                overall_emotion = "positive"
                positive_instance_count += 1

            cv2.putText(input_image, f"Overall: {overall_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            

    cv2.imshow('Emotion Detection', input_image)
    
    #3초 동안 감정 추출하고 루프 종료
    if time.time() - start_time > extract_duration:
        #3초 동안 감지된 n 과 p 감정 횟수를 비교, n 가 더 많으면 count +1
        if negative_instance_count > positive_instance_count:
            count += 1
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total Count : {count}")
