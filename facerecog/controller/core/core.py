import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
# print(tf.__version__)
# print(keras.__version__)

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(96, 96))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

model = keras.models.model_from_json(open("structure.json", "r").read(), custom_objects={'tf': tf})
model.load_weights("openface_weights.h5")
print("Success")

p1 = '14.png'
p2 = '14.png'
img1_representation = model.predict(preprocess_image(p1))[0,:]
img2_representation = model.predict(preprocess_image(p2))[0,:]

cosine = findCosineDistance(img1_representation, img2_representation)
euclidean = findEuclideanDistance(img1_representation, img2_representation)

if cosine <= 0.02:
    print("these are same")
    print(cosine,euclidean)
else:
    print(cosine,euclidean)
    print("these are different")
 
"""if euclidean <= 0.20:
   print("these are same")
else:
   print("these are different")"""