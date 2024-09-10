import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Basic3DConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, include_relu=True):
        super(Basic3DConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.include_relu = include_relu
        if self.include_relu:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.include_relu:
            x = self.relu(x)
        return x

class ResidualIdentityBlock(nn.Module):
    def __init__(self, in_channels, f1, f2):
        super(ResidualIdentityBlock, self).__init__()
        self.block = nn.Sequential(
            Basic3DConvBlock(in_channels, f1, kernel_size=1),
            Basic3DConvBlock(f1, f1, kernel_size=3, padding=1),
            Basic3DConvBlock(f1, f2, kernel_size=1, include_relu=False)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, f2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(f2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out += identity
        return out

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, f1, f2, stride=2):
        super(ResidualConvBlock, self).__init__()
        self.block = nn.Sequential(
            Basic3DConvBlock(in_channels, f1, kernel_size=1, stride=stride),
            Basic3DConvBlock(f1, f1, kernel_size=3, padding=1),
            Basic3DConvBlock(f1, f2, kernel_size=1, include_relu=False)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, f2, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(f2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out += identity
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_classes=10, initial_channels=9, f1_identity=50, f2_identity=50, f1_conv=50, f2_conv=100):
        super(ResNet3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(
            in_channels=initial_channels, 
            identity_f1=f1_identity, 
            identity_f2=f2_identity, 
            conv_f1=f1_conv, 
            conv_f2=f2_conv, 
            num_identity_blocks=3, 
        )
        self.layer2 = self._make_layer(
            in_channels=f2_conv, 
            identity_f1=f1_identity * 2, 
            identity_f2=f2_identity * 2, 
            conv_f1=f1_conv * 2, 
            conv_f2=f2_conv * 2, 
            num_identity_blocks=2, 
        )
        self.layer3 = self._make_layer(
            in_channels=f2_conv * 2, 
            identity_f1=f1_identity * 4, 
            identity_f2=f2_identity * 4, 
            conv_f1=f1_conv * 4, 
            conv_f2=f2_conv * 4, 
            num_identity_blocks=2, 
        )
        self.layer4 = self._make_layer(
            in_channels=f2_conv * 2, 
            identity_f1=f1_identity * 4, 
            identity_f2=f2_identity * 4, 
            conv_f1=0,
            conv_f2=0,
            num_identity_blocks=2, 
            conv_block=False,
        )

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(400, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def _make_layer(self, in_channels, identity_f1, identity_f2, conv_f1, conv_f2, num_identity_blocks, conv_stride=2, conv_block=True):
        layers = []
        for _ in range(num_identity_blocks):
            layers.append(ResidualIdentityBlock(in_channels, identity_f1, identity_f2))
            in_channels = identity_f2  # Update in_channels for the next block
        if conv_block:
            layers.append(ResidualConvBlock(identity_f2, conv_f1, conv_f2, stride=conv_stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

if __name__ == "__main__":

    num_samples = 200
    num_classes = 20
    initial_channels = 9
    image_size = (20, 20, 20)

    X_train = torch.randn(num_samples, initial_channels, *image_size)
    y_train = torch.randint(0, num_classes, (num_samples,))

    model = ResNet3D(num_classes=num_classes, initial_channels=initial_channels, f1_identity=50, f2_identity=50, f1_conv=50, f2_conv=100)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    # Training parameters
    num_epochs = 10
    batch_size = 16

    def train(model, criterion, optimizer, X_train, y_train, num_epochs, batch_size):
        model.train()
        
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            num_batches = len(X_train) // batch_size
            
            for i in range(0, len(X_train), batch_size):
                # Create batch
                inputs = X_train[i:i + batch_size]
                labels = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                if (i // batch_size) % 10 == 9:  # Print every 10 batches
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i // batch_size + 1}/{num_batches}], Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0
        
        print('Finished Training')

    # Call the train function
    train(model, criterion, optimizer, X_train, y_train, num_epochs, batch_size)
