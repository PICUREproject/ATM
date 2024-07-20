import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from methods import load_data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
x_train, y_train = load_data("data/train/")
x_test, y_test = load_data("data/test/")

# 데이터 전처리
x_train1 = x_train / 255
x_test1 = x_test / 255

x_train1 = np.concatenate((x_train1, x_test1))
y_train1 = np.concatenate((y_train, y_test))

x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train1, y_train1, test_size=0.2, shuffle=True, random_state=0)

# 모델 구성
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.22))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['acc'])

print(model.summary())

# 모델 학습
history = model.fit(x_train2, y_train2,
                    batch_size=64,
                    validation_data=(x_test2, y_test2),
                    epochs=40,
                    shuffle=True,
                    callbacks=[callback],
                    verbose=2)

# 학습 정확도 및 손실 그래프 그리기
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 테스트 데이터에 대한 예측
y_pred = model.predict(x_test2)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test2, axis=1)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 분류 보고서 출력
class_report = classification_report(y_true, y_pred_classes)
print('Classification Report:')
print(class_report)
