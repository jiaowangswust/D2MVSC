import torch.nn as nn
import torch.nn.functional as F
#####考虑到，对1x1xC的输入进行1x1的卷积，其效果等同于全连接层，我们可以省去将张量进行降维和升维的过程，于是SE block的实现就变得简单了
# class SE(nn.Module):
#
#     def __init__(self, in_chnls, ratio):
#         super(SE, self).__init__()
#         self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
#         self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
#         self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)
#
#     def forward(self, x):
#         out = self.squeeze(x)
#         out = self.compress(out)
#         out = F.relu(out)
#         out = self.excitation(out)
#         return F.sigmoid(out)
class SE_block(nn.Module):
    def __init__(self, channel_in, reduction):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel_in, channel_in//reduction, bias=False),nn.ReLU(inplace=True), nn.Linear(channel_in//reduction,channel_in, bias= False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ =x.size()

        y = self.avg_pool(x).view(b,c)#view()重构张量的维度
        #print("y avgpool:",y)
        y = self.fc(y).view(b,c,1,1) #expand_as函数将张量扩展为和参数一样的大小
        #print("y fc:",y)
        return x * y.expand_as(x)