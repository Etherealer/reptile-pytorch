from torch import nn


class OmniglotModel(nn.Module):
    def __init__(self, num_classes: int):
        super(OmniglotModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc_layer = nn.Linear(64 * 2 * 2, num_classes)

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        logits = self.fc_layer(out)
        return logits
