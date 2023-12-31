import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import relu

EXPANSION = 4

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, id_downsample=None, stride=1):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * EXPANSION,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * EXPANSION)

        self.relu = relu
        self.id_downsample = id_downsample
    
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.id_downsample is not None:
            identity = self.id_downsample(identity)
        
        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = relu

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Actual Resnet
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * EXPANSION, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
    def _make_layer(self, block, num_blocks, out_channels, stride):
        id_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * EXPANSION: 
            id_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * EXPANSION, 
                                                    kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=out_channels * EXPANSION))
            
        layers.append(block(self.in_channels, out_channels, id_downsample, stride))
        self.in_channels = out_channels * EXPANSION

        for i in range(num_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(CNNBlock, [3, 4, 6, 3], img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(CNNBlock, [3, 4, 23, 3], img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(CNNBlock, [3, 8, 36, 3], img_channels, num_classes)


def test():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(test, batch_size=128,shuffle=False, num_workers=2)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    model = ResNet50(img_channels=1, num_classes=10).to('cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to('cpu'), labels.to('cpu')
            optimizer.zero_grad()
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i%100 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)
            
    print('Training Done')


if __name__ == '__main__':
    test()
