import torch
import torch.nn as nn

class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()
        self.mean = torch.randn(2, 3, 4, 5)
        self.std = torch.randn(2, 3, 4, 5)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=2, stride=3, padding=1, bias=True)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x =  self.conv1(x)
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))


net = JustReshape()
model_name = '../model/just_reshape_0518.onnx'
dummy_input = torch.randn(1, 3, 4, 5)

ouuu = net.forward(dummy_input)
#print(ouuu)
print(ouuu.shape)
#print(ouuu)

dynamic_axes = {'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},'output': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}

## 固定轴
#torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'],dynamic_axes=dynamic_axes)

## 动态轴
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'],dynamic_axes=dynamic_axes)




