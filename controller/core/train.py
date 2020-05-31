import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from model import create_model

import cv2
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
print("Loading pretrained model...")
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
print("Complete!\n")
nn4_small2_pretrained.summary()
print("\n")


import numpy as np
import os.path

limit_image_person = 10
max_person = 36


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
    
def load_metadata(path):
    metadata = []
    print("Loading Data...")
    num = 0
    for i in sorted(os.listdir(path)):
        print("Phase : {}/{}".format(num,len(os.listdir(path))))
        index = 0
        for f in sorted(os.listdir(os.path.join(path, i))):
            index += 1
            if index == limit_image_person:
                break
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext =='.png':
                metadata.append(IdentityMetadata(path, i, f))
        num+=1
        if num == max_person:
            break
    return np.array(metadata)
    print("Done")

# metadata = load_metadata("D:\\lfw")
metadata = load_metadata("D:\\xampp\\htdocs\\python\\OpenCV\\V-CORES\\Face\\SVM\\dataset")

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from align import AlignDlib

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    
    print("enumerating {}/{}".format(i,len(metadata)))
    img = load_image(m.image_path())
    # (w,h) = img.shape[:2]
    # img = cv2.resize(img,(int(w/2),int(h/2)),interpolation=cv2.INTER_LANCZOS4)
    img = align_image(img)
    # scale RGB values to interval [0,1]
    try:
        img = (img / 255.).astype(np.float32)
    except:
        continue
    # obtain embedding vector for image
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
print("Done")



def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

# def show_pair(idx1, idx2):
#     plt.figure(figsize=(8,3))
#     plt.suptitle('Distance = '+str(distance(embedded[idx1], embedded[idx2])))
#     plt.subplot(121)
#     plt.imshow(load_image(metadata[idx1].image_path()))
#     plt.subplot(122)
#     plt.imshow(load_image(metadata[idx2].image_path()));    

# show_pair(77, 78)
# show_pair(77, 50)

from sklearn.metrics import f1_score, accuracy_score

# distances = [] # squared L2 distance between pairs
# identical = [] # 1 if same identity, 0 otherwise

# num = len(metadata)

# for i in range(num - 1):
#     for j in range(i + 1, num):
#         distances.append(distance(embedded[i], embedded[j]))
#         identical.append(1 if metadata[i].name == metadata[j].name else 0)
        
# distances = np.array(distances)
# identical = np.array(identical)

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pickle

targets = np.array([m.name for m in metadata])

print("Fitting Label")
encoder = LabelEncoder()
encoder.fit(targets)
y = encoder.transform(targets)
print("Done")
print("arranging data")
train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

# 50 train examples of 10 identities (5 examples each)
X_train = embedded[train_idx]
# 50 test examples of 10 identities (5 examples each)
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]
print("Done")

param = {'C':[0.01,0.1,1,10,100,500]}

knn = KNeighborsClassifier(weights='distance',n_neighbors=4, metric='euclidean')
# svm = LinearSVC(penalty='l2',loss='squared_hinge',random_state=42)
# print("Creating Model")
clf = GridSearchCV(SVC(class_weight='balanced'), param)
svc = SVC(C=10.0,random_state=100,max_iter=-1, kernel="linear", probability=True,cache_size=200)
# rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)


print("Fitting Model")
clf.fit(X_train,y_train)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
# svm.fit(X_train, y_train)
# rf.fit(X_train,y_train)
print("Done")

print("Calculating Accuracy")
acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svc = accuracy_score(y_test, svc.predict(X_test))
# acc_svm = accuracy_score(y_test, svm.predict(X_test))
# acc_rf = accuracy_score(y_test, rf.predict(X_test))

def export(object,name):
    f = open("trained/"+name, "wb")
    f.write(pickle.dumps(object))
    f.close()

# print('KNN accuracy = ' + str(acc_knn) + ', SVC accuracy = ' + str(acc_svc)+ ', SVM accuracy = ' + str(acc_svm)+ ', Random Forest accuracy = ' + str(acc_rf))
print('KNN accuracy = ' + str(acc_knn)+ ', SVC accuracy = ' + str(acc_svc))
print('Best estimator : '+clf.best_estimator_)
export(svc,"svc-core.vcore")
# export(svm,"svm-core.vcore")
export(knn,"knn-core.vcore")
# export(rf,"rf-core.vcore")
export(y_train,"y_train.vcore")
export(X_train,"X_train.vcore")
print("Sequence Completed")