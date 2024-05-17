import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityShortcut(nn.Module):
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_channels = out_channels - in_channels
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels),'constant',0)
        out = F.max_pool2d(out,1,stride=2)
        # out = F.max_pool2d(out,2)
        return out

class BuildingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # stride is always 1 in middle of building block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding="same", bias=False)   
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()       
        # when dimensions increase, use zero padding  
        if stride != 1 :
            self.shortcut = IdentityShortcut(in_channels,out_channels)
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes= 10 ):
        super(ResNet20, self).__init__()
        # input channels = 16
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(16)

        self.conv2 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes) # input : depth*W*H

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # Downsampling is performed at conv3_1 and conv4_1
        # with a stride of 2
        layers = [block(self.in_channels, out_channels, stride)]
        # change in_channels value 
        # so that input channels of next layer 
        # equal to output channels of prior layer
        self.in_channels = out_channels
        # the other layers in building block have stride of 1
        for i in range(num_blocks-1):
            layers.append(block(self.in_channels, out_channels, 1))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = F.avg_pool2d(out,8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def resnet20():
    return ResNet20(BuildingBlock, [3, 3, 3])
