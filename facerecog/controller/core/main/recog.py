import warnings
from sklearn.svm import LinearSVC
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
# from model import create_model
from align import AlignDlib

encoder = LabelEncoder()

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    

# def load_image(path):
#     img = cv2.imread(path)
#     # OpenCV loads images with color channels
#     # in BGR order. So we need to reverse them
#     return img

alignment = AlignDlib('../models/landmarks.dat')
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# nn4_small2_pretrained = create_model()
# nn4_small2_pretrained.load_weights('../weights/nn4.small2.v1.h5')

vid = cv2.VideoCapture(0)
metadata = pickle.loads(open("metadata.vcore", "rb").read())
targets = np.array([m.name for m in metadata])
# y = encoder.transform(targets)
# embedded = np.zeros((metadata.shape[0], 128))
X_train = pickle.loads(open("x_train.vcore","rb").read())
Y_train = pickle.loads(open("y_train.vcore","rb").read())

svm = pickle.loads(open("svm-core.vcore", "rb").read())
protoPath = os.path.join("../models", "deploy.prototxt")
modelPath = os.path.join("../models",
    "res10_300x300_ssd_iter_140000.caffemodel")

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("../weights/openface_nn4.small2.v1.t7")

names = ["Abdurrahman","Abid Fakhri","Ahmad Saugi","Lexi Anugrah"]

if __name__ == "__main__":
    # print(metadata)
    while True:
        _,frame = vid.read()
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                aligned = align_image(frame)
                # aligned = np.expand_dims(aligned, axis=0)
                # img = (aligned / 255.).astype(np.float32)
                try:
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),(0, 0, 0), swapRB=True, crop=False)
                except:
                    continue
                embedder.setInput(faceBlob) 
                # embeddings = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
                prediction = svm.predict_proba(embedder.forward())[0]
                similar = np.argmax(prediction)
                # print(similar)
                name = names[similar]
                proba = prediction[similar]
                # example_prediction = svc.predict(embeddings)
                # example_identity = encoder.inverse_transform(example_prediction)[0]

                cv2.imshow("window",frame)
                print('Recognized as ' + str(name), proba)
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    vid.release()
                    cv2.destroyAllWindows()
                    break