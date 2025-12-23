import torch
from torch import nn
import torch.nn.functional as F


class DynamicLayer(nn.Module):
    # 动态神经网络结构，参数：输入维度、输出维度、最大深度
    def __init__(self, input_dim, output_dim, max_depth):
        super(DynamicLayer, self).__init__()
        self.max_depth = max_depth
        self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else output_dim, output_dim) for i in range(max_depth)])
        self.gates = nn.ModuleList([nn.Linear(output_dim, 1) for _ in range(max_depth)])

    def forward(self, x, depth=0):
        #根据最大深度生成多个线性层
        if depth >= self.max_depth:
            return x
        #通过当前深度的线性层之后，经relu激活函数处理
        x = F.relu(self.layers[depth](x))
        #使用sigmoid函数计算当前层的门控值
        gate_status = torch.sigmoid(self.gates[depth](x)).mean()
        #若门控值超过预设阈值，则网络递归调用自身进入下一层
        if gate_status > 0.5:
            return self.forward(x, depth + 1)
        else:
            return x


