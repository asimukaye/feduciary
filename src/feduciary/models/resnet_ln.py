import torch

from feduciary.models.model_utils import ResidualBlock, BottleneckBlockLN, ResidualBlockLN



__all__ = ['ResNet10LN', 'ResNet18LN', 'ResNet34LN']

CONFIGS = {
    'ResNet10': [1, 1, 1, 1],
    'ResNet18': [2, 2, 2, 2],
    'ResNet34': [3, 4, 6, 3],
    'ResNet50': [3, 4, 6, 3],
}

class ResNetLN(torch.nn.Module):
    def __init__(self, config, block, in_channels, hidden_size, num_classes):
        super(ResNetLN, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(True),
            self._make_layers(block, 64, config[0], stride=1),
            self._make_layers(block, 128, config[1], stride=2),
            self._make_layers(block, 256, config[2], stride=2),
            self._make_layers(block, 512, config[3], stride=2),
        ) 
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear((7 * 7) * 512, self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.hidden_size, planes, stride))
            self.hidden_size = planes
        return torch.nn.Sequential(*layers)

class ResNet10LN(ResNetLN):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet10LN, self).__init__(CONFIGS['ResNet10'], ResidualBlock, in_channels, hidden_size, num_classes)

class ResNet18LN(ResNetLN):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet18LN, self).__init__(CONFIGS['ResNet18'], ResidualBlock, in_channels, hidden_size, num_classes)

class ResNet34LN(ResNetLN):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet34LN, self).__init__(CONFIGS['ResNet34'], ResidualBlock, in_channels, hidden_size, num_classes)

class ResNet50LN(ResNetLN):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(ResNet50LN, self).__init__(CONFIGS['ResNet50'], BottleneckBlockLN, in_channels, hidden_size, num_classes)
