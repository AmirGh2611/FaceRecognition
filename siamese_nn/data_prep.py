from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


def reading_data(path, t):
    data = ImageFolder(path, transform=t)
    # print(len(data))
    return data  # return Dataset obj


images_path = r"C:\Users\amirhossein\Desktop\FaceRecognition\data"
transform = transforms.Compose([transforms.ToTensor()])

dataset = reading_data(images_path, transform)

train_set, valid_set, test_set = random_split(dataset, [1200, 500, 100])  # random only accepts Dataset obj

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True)
