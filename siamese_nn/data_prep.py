import torch
import cv2
from torch.utils.data import TensorDataset, DataLoader, random_split
import random
import glob
import numpy as np


def dataset_and_split(x, y):
    x = torch.tensor(np.array(x)) # list --> numpy --> torch is faster than list --> torch
    y = torch.tensor(np.array(y))
    y = y.unsqueeze(dim=1)  # (5000) --> (5000,1)
    dataset = TensorDataset(x, y)
    train, validation, test = random_split(dataset, [3500, 1000, 500])
    return train, validation, test


def reading_data(path):
    images = []
    labels = []
    for address in glob.glob(path):
        img = cv2.imread(address)
        img = image_preprocessing(img)
        images.append(img)
        labels.append(address.split("\\")[-2])
    return images, labels  # list


def image_preprocessing(image):
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))  # torch array is channel first
    image = image.astype("float32") / 255.0  # image normalization
    return image


def make_pairs(images, labels, num_pairs=5000):
    """
    Creates balanced positive/negative pairs from a Dataset (e.g. train_set)
    Returns list of tuples: (img1, img2, label)
    """
    label_to_indices = {}
    for i, label in enumerate(labels):  # classifying image indexes with same label
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)

    classes = list(label_to_indices.keys())
    pairs = []
    targets = []

    while len(pairs) < num_pairs:
        if random.random() < 0.5:
            # Positive pair
            cls = random.choice(classes)  # choosing a class
            i1, i2 = random.sample(label_to_indices[cls], 2)  # find two images from same class
            targets.append(1)
        else:
            # Negative pair
            cls1, cls2 = random.sample(classes, 2)  # choosing two different classes
            i1 = random.choice(label_to_indices[cls1])  # find two images from two different classe
            i2 = random.choice(label_to_indices[cls2])
            targets.append(0)

        pairs.append((images[i1], images[i2]))
    return pairs, targets  # list


images_path = r"C:\Users\amirhossein\Desktop\FaceRecognition\data\*\*.jpg"
X, Y = reading_data(images_path)
image_pairs, labels = make_pairs(X, Y)
train_set, validation_set, test_set = dataset_and_split(image_pairs, labels)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
valid_loader = DataLoader(validation_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True)
