import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from torch.utils.data import DataLoader, Dataset


class ImagePrep:
    def __init__(self, image_path):
        self.images = []
        self.labels = []
        self.image_path = image_path
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def read_image(self):
        for address in glob.glob(self.image_path):
            img = cv2.imread(address)
            img = ImagePrep.normalizer(img)
            self.images.append(img)
            self.labels.append(address.split("\\")[-2])
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    @staticmethod
    def normalizer(img):
        img = cv2.resize(img, (256, 256))
        img = np.transpose(img, (2, 0, 1))
        img = img.astype("float32") / 255.0
        return img

    def data_splitter(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.images, self.labels,
                                                                                test_size=0.2, random_state=42)

    @staticmethod
    def make_pair(x, y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = max(y) + 1
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

        pairs = []
        labels = []

        for idx1 in range(len(x)):
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [1]

            label2 = random.randint(0, num_classes - 1)
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [0]

        return np.array(pairs), np.array(labels)

    def data_loader(self):
        pairs_train, labels_train = obj.make_pair(self.x_train, self.y_train)
        pairs_test, labels_test = obj.make_pair(self.x_test, self.y_test)
        train_dataset = Dataset(pairs_train, labels_train)
        test_dataset = Dataset(pairs_test, labels_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        return train_loader, test_loader


obj = ImagePrep(r"C:\Users\amirhossein\Desktop\FaceRecognition\data\*\*.jpg")
obj.read_image()
obj.data_splitter()
train_loader, test_loader = obj.data_loader()
