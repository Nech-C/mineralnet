import torch.nn as nn
import torchvision.models as models


class MineralClassifierRes18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


