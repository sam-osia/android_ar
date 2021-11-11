import numpy as np
import cv2
from utils import *


set_path()

raw_data_dir = './data/raw'
processed_data_dir = './data/processed'

rgb_3_channel_data = []
rgb_1_channel_data = []
depth_3_channel_data = []
depth_1_channel_data = []

rgb_label = []
depth_label = []


for file in os.listdir(raw_data_dir):
    print(file)
    file_name = file.split('.')[0]
    type = file_name[:-1]
    label = int(file_name[-1])

    cap = cv2.VideoCapture(os.path.join(raw_data_dir, file))
    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if type == 'depth':
                depth_3_channel_data.append(frame)
                depth_1_channel_data.append(gray)
                depth_label.append(label)
            elif type == 'rgb':
                rgb_3_channel_data.append(frame)
                rgb_1_channel_data.append(gray)
                rgb_label.append(label)
            else:
                print("Type not recognized!")

        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


rgb_3_channel_data = np.array(rgb_3_channel_data)
rgb_1_channel_data = np.array(rgb_1_channel_data)
depth_3_channel_data = np.array(depth_3_channel_data)
depth_1_channel_data = np.array(depth_1_channel_data)

rgb_label = np.array(rgb_label)
depth_label = np.array(depth_label)

np.save(os.path.join(processed_data_dir, 'rgb_3_channel_data'), rgb_3_channel_data)
np.save(os.path.join(processed_data_dir, 'rgb_1_channel_data'), rgb_1_channel_data)
np.save(os.path.join(processed_data_dir, 'depth_3_channel_data'), depth_3_channel_data)
np.save(os.path.join(processed_data_dir, 'depth_1_channel_data'), depth_1_channel_data)

np.save(os.path.join(processed_data_dir, 'rgb_label'), rgb_label)
np.save(os.path.join(processed_data_dir, 'depth_label'), depth_label)
