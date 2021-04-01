import torch
import torch.nn as nn
import numpy as np

class ThreeDFCNN(nn.Module):
    def __init__(self):
        super(ThreeDFCNN, self).__init__()

        self.f3d_1 = nn.Conv3d(1, 64, kernel_size=(7, 9, 9), stride=1, padding=(0, 0, 0)) #64表示filter number
        #对于3d卷积来说，filter number不能再理解为通常意义上的通道数了。
        #输入图像的通道数被band num所取代
        #3D卷积相对于2D卷积最大的优势就是，如果我愿意，我可以让输入输出之间的bandnum保持不变。
        #3D卷积多出来的哪一维可以为光谱波段之间的关系建模。

        self.f3d_2 = nn.Conv3d(64, 32, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.f3d_3 = nn.Conv3d(32, 9, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.f3d_4 = nn.Conv3d(9, 1, kernel_size=(3, 5, 5), stride=1, padding=(0, 0, 0))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.f3d_1(x))
        x = self.relu(self.f3d_2(x))
        x = self.relu(self.f3d_3(x))
        x = self.f3d_4(x)
        return x

"""
batch_size = 10
band_num = 111
img_width = 33
img_height = 33
input_chl_num = 1 #对于高光谱图像来说，这个可以指定为1,最初始的图像，我们可以将其看成有一维特性。

#mat = torch.randn(batch_size, 1, 111, 33, 33)
mat = torch.randn(batch_size, input_chl_num, band_num, img_width, img_height)
net = ThreeDFCNN()
print(net(mat).shape) #torch.Size([10, 1, 103, 21, 21]) （batch_size, input_channel, band_num, width, height）
#从结果上看 band_num(103) = 111 - 8
"""