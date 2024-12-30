import os
import cv2
import numpy as np

dataset_path = r"C:\Users\nates\Documents\FACE_REC\DATA"
faceData = []
labels = []
classId = 0
nameMap = {}

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):

        nameMap[classId] = f[:-4]
        #storing X-values
        dataItem = np.load(os.path.join(dataset_path, f))  # Load the .npy file
        m = dataItem.shape[0]
        faceData.append(dataItem)

        #creating Y-values
        target = classId * np.ones((m, ))
        classId+=1
        labels.append(target)

#Creating the dataset out of the obtained data
XT = np.concatenate(faceData, axis=0)
yT = np.concatenate(labels,axis=0).reshape((-1, 1))