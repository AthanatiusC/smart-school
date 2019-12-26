import warnings
import pickle

import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

if __name__ == "__main__":
    metadata = pickle.loads(open("metadata.vcore", "rb").read())
    svc = pickle.loads(open("svc-core.vcore", "rb").read())
    embedded = pickle.loads(open("embedding.vcore", "rb").read())
    targets = np.array([m.name for m in metadata])

    encoder = LabelEncoder()
    encoder.fit(targets)

    # Suppress LabelEncoder warning
    warnings.filterwarnings('ignore')

    example_idx = 29

    test_idx = np.arange(metadata.shape[0]) % 2 == 0

    example_image = load_image(metadata[test_idx][20].image_path())
    example_prediction = svc.predict([embedded[test_idx][20]])
    example_identity = encoder.inverse_transform(example_prediction)[0]

    cv2.imshow('Recognized as '+example_identity,example_image)