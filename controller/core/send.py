import warnings
from sklearn.svm import LinearSVC
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
from align import AlignDlib
import requests
import json

alignment = AlignDlib('models\\landmarks.dat')
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

vid = cv2.VideoCapture(0)

protoPath = os.path.join("models", "deploy.prototxt")
modelPath = os.path.join("models",
    "res10_300x300_ssd_iter_140000.caffemodel")

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("weights/openface_nn4.small2.v1.t7")

user_id = input("User ID : ")
embeddings = []

if __name__ == "__main__":
    # print(metadata)
    while True:
        _,frame = vid.read()
        try:
            (h, w) = frame.shape[:2]
        except:
            continue
        imageBlob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                y = startY - 10 if startY - 10 > 10 else startY + 10
                try:
                    face = frame[startY-10:endY+10, startX-10:endX+10]
                    (h,w) =face.shape[:2]
                    aligned = align_image(face)
                    faceBlob = cv2.dnn.blobFromImage(aligned, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    if cv2.waitKey(1)==ord('c'):
                        embeddings.append(np.array(vec[0]).astype("float64").tolist())
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 4)
                except Exception as e:
                    print(e)
                    pass              
        cv2.imshow("window", frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27 or len(embeddings) == 20:
            vid.release()
            cv2.destroyAllWindows()
            break
print(embeddings[0][0])
r = requests.post(url="http://localhost:8088/api/v2/user/verify",json= {"user_id":int(user_id),"embeddings":np.array(embeddings).astype("float64").tolist()})
res = r.json()
print(res)
