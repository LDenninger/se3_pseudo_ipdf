import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, depth=50, layer=0, pretrained=True):
        super(ResNet, self).__init__()
        assert depth in [18, 34, 50]
        resnet_dict = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50
        }
        resnet = resnet_dict[depth](pretrained=pretrained)
        # remove last FC layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-(1+layer)])

    def forward(self, input_):
        out = self.resnet(input_)
        out = out.view(out.shape[0], -1)
        return out

