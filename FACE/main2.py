import tensorflow as tf

print(f"TensorFlow version in face_detection_yolo.py: {tf.__version__}")

import cv2
import time
from ultralytics import YOLO

# 클래스 ID와 클래스 이름을 매핑하는 딕셔너리
class_names = {0: 'ear', 1: 'eye', 2: 'mouth', 3: 'nose'}

# YOLOv8 모델 로드
model = YOLO('best.pt')  

# 웹캠 초기화
cap = cv2.VideoCapture(0)  

count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 모델 예측
    results = model(frame, imgsz=640, conf=0.6)

    # 결과에서 감지된 객체 확인
    detected_classes = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = class_names.get(class_id, 'unknown')
        detected_classes.append(class_name)

    detected_classes_set = set(detected_classes)  # 감지된 클래스들을 세트로 저장

    # 눈, 코, 입 모두 감지되었는지 확인
    required_classes = {'eye', 'nose', 'mouth'}
    confirmed = required_classes.issubset(detected_classes_set)  # 세트 비교를 통해 확인

    # confirmed가 True일 때 프로그램 종료
    if confirmed:
        print("Face Confirmed")
        cv2.putText(frame, 'Confirmed', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('YOLOv8 Webcam', frame)
        cv2.waitKey(2000)  # 2초 대기하여 메시지 확인
        break
    else:
        # 2초 동안 감지되지 않았을 경우
        if time.time() - start_time >= 2:
            count += 1
            start_time = time.time()  # 타이머 초기화
            print(f"Face not detected for {count} times")
            cv2.putText(frame, 'Not Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            
            # count가 2보다 크면 프로그램 종료
            if count > 2:
                print("Face not detected")
                break

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 출력 프레임 보여주기
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
