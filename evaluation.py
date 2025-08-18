import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import ImageOps, ImageFile
import numpy as np

# Handle Truncated Images
ImageFile.LOAD_TRUNCATED_IMAGES =True

# There is not training as this is a web application. So I set device to cpu
device = 'cpu' 

# Hyper paramaters
batch_size = 1

# Model Path
PATH = './saved_model.pth'

# Resize and Tranform Images
class ResizeWithPadding:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, image):
        return ImageOps.pad(image, self.target_size, color=(0,0,0))

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128,256, 3)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.conv5 = nn.Conv2d(512, 1024, 3)
        self.conv6 = nn.Conv2d(1024, 2048, 3)
        self.conv7 = nn.Conv2d(2048, 4096, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 3)
        self.dropout = nn.Dropout(.4)

    def forward(self, x):
        # N, 3, 300, 300 (batch_size, number of colors, pixel_length, pixel_height)
        x = F.leaky_relu(self.conv1(x))   # N, 128, 298, 298   
        x = self.pool(x)            # N, 128, 149, 149
        x = F.leaky_relu(self.conv2(x))   # N, 256, 147, 147
        x = self.pool(x)            # N, 256, 73, 73
        x = F.leaky_relu(self.conv3(x))   # N, 512, 71, 71
        x = self.pool(x)            # N, 512, 35, 35
        x = F.leaky_relu(self.conv4(x))   # N, 1024, 33, 33
        x = self.pool(x)            # N, 1024, 16, 16
        x = F.leaky_relu(self.conv5(x))   # N, 2048, 14, 14
        #x = self.pool(x)            # N, 2048, 7, 7
        x = F.leaky_relu(self.conv6(x))   # N, 4096, 12, 12
        '''x = self.pool(x)            # N, 8192, 2, 2'''
        #x = F.relu(self.conv7(x))   # N, 8192, 27, 27
        x = self.gap(x)             # N, 8192, 1, 1
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))     # N, 512
        x = self.dropout(x)
        x = self.fc2(x)             # N, 3
        return x

def evaluate(test):

    transform = transforms.Compose([
        ResizeWithPadding((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # test = 'test-data-set/'

    test_dataset = torchvision.datasets.ImageFolder(root=test, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = test_dataset.classes

    loaded_model = ConvNet()
    loaded_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    loaded_model.to(device)
    loaded_model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)
        results = []
        confidence_scores = []

        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = loaded_model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            #max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

            for j in range(images.size(0)):
                    print(f"Predicted index: {predicted[j].item()}, Class names: {class_names}")
                    true_label = class_names[labels[j].item()]
                    pred_label = class_names[predicted[j].item()]
                    results.append([i * batch_size + j, true_label, pred_label])
                    class_probs = probs[j] * 100
                    confidence_scores.append(class_probs)

        acc = 100.0 * n_correct / n_samples

        print('Accuracy is ', acc, '%')


        np_results = np.array(results)
        print(np_results)
        print(confidence_scores)
        user_prediction = np_results[3,2]
        user_confidences = confidence_scores[3]
        predictions = {
            class_names[0]: float(user_confidences[0]),
            class_names[1]: float(user_confidences[1]),
            class_names[2]: float(user_confidences[2])
            }
        return [{"label": "cat", "confidence": predictions["cat"]},
                {"label": "dog", "confidence": predictions["dog"]},
                {"label": "other", "confidence": predictions["other"]}]