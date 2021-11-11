import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import cv2
from utils import *


def normalize_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, dsize=(160, 120), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(float) - 128) / 128
    return normalized

set_path()

model = keras.models.load_model('./models/first_try')

predict_dir = './data/predict'
video_name = 'depthrandom2.mp4'

cap = cv2.VideoCapture(os.path.join(predict_dir, video_name))
if not cap.isOpened():
    print("Error opening video stream or file")

full_frames = []
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        normalized = normalize_frame(frame)
        full_frames.append(frame)
        frames.append(normalized)
    else:
        break

frames = np.array(frames)
print(frames.shape)
predictions = model.predict(frames)
print(predictions.shape)
print(predictions)

predict_labels = np.argmax(predictions, axis=1)
unique, counts = np.unique(predict_labels, return_counts=True)
print('predicted label count:', dict(zip(unique, counts)))

video_writer = cv2.VideoWriter(os.path.join(predict_dir, f'{video_name.split(".")[0]}_labeled.avi'),
                               cv2.VideoWriter_fourcc(*'XVID'), 25.0, (320, 240))

for i, f in enumerate(full_frames):
    cv2.putText(f, str(np.argmax(predictions[i])), (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow('window', f)
    cv2.waitKey(1)
    video_writer.write(f)
cv2.destroyAllWindows()

print('done')
video_writer.release()
