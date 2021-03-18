import torch.nn as nn
import torch

from utils.my_logging import My_Logging as My_Logging
from . seq2seq import BasicBlock as BasicBlock

from typing import List
Vector = List[int]

# 定义日志配置实例
my_log = My_Logging()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")



class Stack(nn.Module):
    def __init__(self, input_size:int=2, 
                        hidden_size_list:Vector=[4,5,6,7,8], 
                        input_seqlen:int=14, 
                        forecast_seqlen:int=3):
        super(Stack,self).__init__()
        
        self.estimate_seqlen = input_seqlen
        self.forecast_seqlen = forecast_seqlen

        self.input_size = input_size
        self.hidden_size_list = hidden_size_list


        self.block_1 = BasicBlock(  input_size=self.input_size, 
                                    hidden_size=self.hidden_size_list[0],
                                    forecast_seqlen=self.forecast_seqlen,
                                    estimate_seqlen=self.estimate_seqlen) 
                                    
        self.block_2 = BasicBlock(  input_size=1, 
                                    hidden_size=self.hidden_size_list[1],
                                    forecast_seqlen=self.forecast_seqlen,
                                    estimate_seqlen=self.estimate_seqlen) 
        '''                                                        
        # 使用Sequential的方法
        blocks = []
        self.num_blocks = len(hidden_size_list)
        self.my_sequential = My_Sequential(forecast_seqlen=self.forecast_seqlen)
        
        for i in range(self.num_blocks):
            input_size = 1 if i != 0 else self.input_size
            blocks += [BasicBlock(input_size=input_size, 
                                    hidden_size=self.hidden_size_list[i],
                                    forecast_seqlen=self.forecast_seqlen,
                                    estimate_seqlen=self.estimate_seqlen) ]
        self.net = self.my_sequential(*blocks)
        '''
        
    
    def forward(self, inputs):
        '''
        params:
            inputs : 整个stack的输入 [batch_size, seq_len, feature=2]
        returns:
            forecast : 所有basicblock的预测输出之和  [batch_size, predict_seqlen, 1]
            input_estimate: 模型最后的输出，也就是输入 减  最佳估计 [batch_size, input_seqlen, 1]
        '''
        # forcast value ，之后的每一个block输出的预测值都加进来
        batch_size = inputs.shape[0]
        forecast = torch.zeros(size=(batch_size, self.forecast_seqlen, 1)).to(DEVICE)   
        
        # forecast_ 和 x_ 都是临时变量 暂时存放当前 block 的输出
        forecast_, x_ = self.block_1(inputs)
        forecast +=forecast_
        inputs = x_
    
        forecast_, x_ = self.block_2(inputs)
        forecast +=forecast_
        inputs = x_
          
        '''
        # 使用Sequential方法
        forecast, input_estimate = self.net(inputs)
        '''
        return forecast, inputs



if __name__ == "__main__":
    my_log.info_logger('begin')
    batch_size = 128
    time_step = 14
    features = 2
    # 测试 输入为 (batch_size, time_step, features) 是否可行
    x = torch.rand(size =(batch_size, time_step, features)).to(DEVICE)
 
    model = Stack(input_size=features, hidden_size_list=[5,6], input_seqlen=time_step, forecast_seqlen=7).to(DEVICE)
    y, e = model(x)
    print(y.shape)
    print(e.shape)