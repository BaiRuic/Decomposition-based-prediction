import torch
import torch.nn as nn
import logging
from torch.nn.utils import weight_norm




logging.basicConfig(filename='log.log',        
                    filemode='a',              
                    level=logging.INFO,         
                    format="%(asctime)s--%(lineno)d   %(message)s",   
					          datefmt= "%Y-%m-%d  %H:%M:%S %a")  

logging.info("-------Start print log--------")


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        '''裁剪多出来的时间步
        param:
            x shape: [batch_size, features, seq_len]
        return :
            shape: [batch_size, features, seq_len-self.chomp_size]
        '''
        
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # first part
        # 输出为 [batch_size, n_outputs, seq_len+padding]
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 输出为 [batch_size, n_outputs, seq_len]
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # second part
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # 1*1 的卷积
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
      '''
      numn_inputs: 原始数据输入通道数
      num_channels: len(num_channels) 个block ，第i个boock特征维数为num_channels[i]
                    例如[25,25,25,25]表示有4个block，每层block里面序列数据特征数为25
      kernel_size: 卷积核尺寸
      '''
        super(TemporalConvNet, self).__init__()
        layers = []
        # 有 num_levels 个 block
        num_levels = len(num_channels) 
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                        stride=1, dilation=dilation_size,
                        padding=(kernel_size-1) * dilation_size, 
                        dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        '''
        param:
            x: the shape of x [batch, input_channel, seq_len]
        return:
            shape:[batch, output_channel, seq_len]
        '''
        return self.network(x)
'''
num_inputs = 2
num_channel = [3, 5, 8]
kernel_size = 5

i = 0
    dilation_size = 2^0 = 1
    in_channels = num_inputs=2
    out_channels = num_channels[0] =3
    TemporalBlock(2,3,
                kernel_size=5,
                stride=1, 
                dilation=dilation_size=1
                padding = 4

i = 1
    dilation_size = 2^1 = 2
    in_channels = num_channel[0]=3
    out_channels=num_channel[1] = 5
    TemporalBlock(3,5,
                kernel_size=5
                stride=1
                dilation=dilation_size=2
                padding=8

'''

model = nn.Conv1d(2,3,5,1,4,1)
# model = TemporalBlock(n_inputs=2, n_outputs=3, kernel_size=5, stride=1, dilation=1, padding=4)

batch_size = 128
seq_len = 748
channels = 2
x = torch.rand(size=(batch_size, channels, seq_len))
y = model(x)
logging.info(model)
logging.info(f"{x.shape}{y.shape}")