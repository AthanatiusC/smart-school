import warnings
from sklearn.svm import LinearSVC
import pickle
import os
import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
from align import AlignDlib
# from tensorflow.keras.models import load_model

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

alignment = AlignDlib('models/landmarks.dat')
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

try:
    vid = cv2.VideoCapture(0)
    # vid = cv2.VideoCapture("rtsp://admin:AWPZEO@192.168.1.64/0/h264_stream")

    # model = pickle.loads(open("trained/svc-core.vcore", "rb").read())
    # model = load_model ("models\\trained.h5")
    # model.summary()
    protoPath = os.path.join("models", "deploy.prototxt")
    modelPath = os.path.join("models",
        "res10_300x300_ssd_iter_140000.caffemodel")

    recognizer = pickle.loads(open("trained/face_verification.pkl", "rb").read())
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embedder = cv2.dnn.readNetFromTorch("weights/openface_nn4.small2.v1.t7")
except Exception as e:
    print(e)
# names = ["Abdurrahman","Abid Fakhri","Ahmad Saugi","Lexi Anugrah"]
names = []
for name in os.listdir("D:\\xampp\\htdocs\\python\\OpenCV\\V-CORES\\Face\\SVM\\dataset"):
    names.append(name)

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
                    face = frame[startY-100:endY+100, startX-100:endX+100]
                    (h,w) =face.shape[:2]
                    # face = cv2.resize(face,(int(w/2),int(h/2)),interpolation=cv2.INTER_LANCZOS4)
                    aligned = align_image(face)
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = names[j]
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    if int(proba * 100) > 50:
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 4)
                        text = name+" "+str(int(proba*100))+"%"
                        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    else:
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 4)
                        text = "Unknown"
                        cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                except:
                    continue
        cv2.imshow("window",frame)
        # print('Recognized as ' + str(name), proba)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            vid.release()
            cv2.destroyAllWindows()
            break