import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import ImageOps, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES =True

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper paramaters
num_epochs = 90
batch_size = 16
learning_rate = 0.00004

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
        x = F.relu(self.conv1(x))   # N, 128, 298, 298   
        x = self.pool(x)            # N, 128, 149, 149
        x = F.relu(self.conv2(x))   # N, 256, 147, 147
        x = self.pool(x)            # N, 256, 73, 73
        x = F.relu(self.conv3(x))   # N, 512, 71, 71
        x = self.pool(x)            # N, 512, 35, 35
        x = F.relu(self.conv4(x))   # N, 1024, 33, 33
        #x = self.pool(x)            # N, 1024, 16, 16
        x = F.relu(self.conv5(x))   # N, 2048, 31, 31
        #x = self.pool(x)            # N, 2048, 7, 7
        x = F.relu(self.conv6(x))   # N, 4096, 29, 29
        '''x = self.pool(x)            # N, 8192, 2, 2'''
        #x = F.relu(self.conv7(x))   # N, 8192, 27, 27
        x = self.gap(x)             # N, 8192, 1, 1
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))     # N, 512
        x = self.dropout(x)
        x = self.fc2(x)             # N, 3
        return x

def train():

    transform = transforms.Compose([
        ResizeWithPadding((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = torchvision.datasets.ImageFolder(root='cats-vs-dogs/',
                                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=6,persistent_workers=True, pin_memory=True, shuffle=True)

    
            
    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)

    for epoch in range(num_epochs):
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f'Epoch Training', leave=False, ncols=80)

        for i, (images, labels) in enumerate(train_loader):

            # send to gpu
            images = images.to(device)
            labels = labels.to(device)

            # forward pass (make predication and calc loss)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass (calc gradients and update weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #scheduler.step()

            running_loss += loss.item()

            loop.update()

        loop.close()
        print(f"Epoch[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}")

    print('finished training')
    PATH = './saved_model.pth'
    torch.save(model.state_dict(), PATH)
    
PATH = './saved_model.pth'
def evaluate():

    transform = transforms.Compose([
        ResizeWithPadding((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    test_dataset = torchvision.datasets.ImageFolder(root='test-data-set/', transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loaded_model = ConvNet()
    loaded_model.load_state_dict(torch.load(PATH))
    loaded_model.to(device)
    loaded_model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = loaded_model(images)

            #max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples

        print('Accuracy is ', acc, '%')

if __name__ == '__main__':
    train()
    evaluate()