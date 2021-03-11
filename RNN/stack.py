import torch
import torch.nn as nn
import sys
sys.path.append("..")

from utils.my_logging import My_Logging as My_Logging
from typing import List

Vector = List[int]

# 定义日志配置实例
my_log = My_Logging()

class LstmEncoder(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        '''
        input_size:    the number of features in the input X
        hidden_size:   the number of features in the hidden state h
        '''
        super(LstmEncoder,self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)
    def forward(self, inputs):
        '''
        param:
            inputs [batch_size, time_steps, features]
                    function： 作为编码器的输入
        return:
            hidden [1, batch_size, hidden_size]
                    function: 作为编码器的输出，只包含LSTM的最后状态
            cell: [1, batch_size, hidden_size]
        '''
        output, (hidden, cell) = self.lstm(inputs)
        return hidden, cell

class LstmDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(LstmDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=True)

    def forward(self, inputs, prev_hidden, prev_cell):
        '''
        params:
            inputs: [batch_size, seq_len=1, features=1]
            prev_hidden [1, batch_size, hidden_size]
            prev_cell [1, batch_size, hidden_size]
        returns:
            prediction: [batc_size, seq_len=1, feature=1]
            hidden: [1, batch_size, hidden_size]
            cell: [1, batch_size, hidden_size]
        '''
        # output [batch_size, seq_len=1, features=hidden_size]
        # hidden [1, batch_size, hidden_size]
        # cell [1, batch_size, hidden_size]
        output, (hidden, cell) =  self.lstm(inputs, (prev_hidden, prev_cell))
        # prediction [batch_size, seq_len=1, output_size=1]
        prediction = self.fc(output.squeeze(dim=1)).unsqueeze(dim=1)
        return prediction, hidden,cell

class BasicBlock(nn.Module):
    def __init__(self,input_size, hidden_size, forecast_seqlen, estimate_seqlen):
        '''
        input_size: 输出序列数据的样本特征
        hidden_size: 隐藏层样本特征
        predice_seqlen: 预测的时间步
        estimate_seqlen: 估计的时间步，即输入样本的时间步
        '''
        super(BasicBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forecast_seqlen = forecast_seqlen
        self.estimate_seqlen = estimate_seqlen
        
        self.encoder = LstmEncoder(input_size=self.input_size, hidden_size=self.hidden_size)
        self.f_decoder = LstmDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)
        self.e_decoder = LstmDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)

    def forward(self, inputs):
        
        batch_size = inputs.shape[0]

        # 对输入进行编码 得到编码状态
        hidden, cell = self.encoder(inputs)

        my_log.info_logger(f"Encoder_input:  inputs:{inputs.shape}")
        my_log.info_logger(f"Encoder_output: hidden:{hidden.shape}")
        my_log.info_logger(f"Encoder_output: cell:{cell.shape}")
        assert(hidden.shape==(1, batch_size, self.hidden_size))
        assert(cell.shape==(1, batch_size, self.hidden_size))

        # 解码器输入
        f_decoder_input = inputs[:, -1, 0].reshape(batch_size, 1, 1)  # for forecast decoder
        e_decoder_input = inputs[:, -1, 0].reshape(batch_size, 1, 1)  # for estimate decoder

        my_log.info_logger(f"f_Dncoder_input:  f_decoder_input:{f_decoder_input.shape}")
        my_log.info_logger(f"e_Dncoder_input:  e_decoder_input:{e_decoder_input.shape}")

        assert(f_decoder_input.shape==(batch_size, 1, 1))
        assert(e_decoder_input.shape==(batch_size, 1, 1))

        # 存放 两个解码器的输出列表
        forecast_outputs = []  
        estimate_outputs = []  

        # 初始化 解码器状态
        f_hidden, f_cell = hidden, cell   # for forecast decoder
        e_hidden, e_cell = hidden, cell   # for estimate decoder

        # 做预测
        for _ in range(self.forecast_seqlen):
            out, f_hidden,f_cell = self.f_decoder(f_decoder_input, f_hidden, f_cell)
            forecast_outputs.append(out)
            f_decoder_input = out

        # 做估计
        for _ in range(self.estimate_seqlen):
            out, e_hidden, e_cell = self.e_decoder(e_decoder_input, e_hidden, e_cell)
            estimate_outputs.append(out)    
            e_decoder_input = out
        
        # 将列表 estimate_outputs 和 forecast_outputs 转换为tensor
        estimate_outputs = torch.cat(estimate_outputs, dim=1)  # [batch_size, seq_len=estimate_seqlen, 1]
        forecast_outputs = torch.cat(forecast_outputs, dim=1)  # [batch_size, seq_len=estimate_seqlen, 1]

        # 提取输入数据中的 需要预测变量数据 如负载预测中，提取负载数据。剔除温度数据
        input_main = inputs[:,:,0].unsqueeze(dim=2) # [batch_size, seq_len, features=1]
        my_log.info_logger(f"BasicBlock_output:input_main:{input_main.shape}")
        my_log.info_logger(f"BasicBlock_output:estimate_outputs:{estimate_outputs.shape}")
        my_log.info_logger(f"BasicBlock_output:forecast_outputs:{forecast_outputs.shape}")
        assert(input_main.shape == estimate_outputs.shape)

        return forecast_outputs, input_main-estimate_outputs

class My_Sequential(nn.Sequential):
    '''
    nn.Sequential 里面的module都是单输入单输出的，但是我的 BasicBlock 是单输入，两输出
    所以继承nn.Sequential,且重写foward函数 使得输出支持多输出 
    '''
    def __init__(self, forecast_seqlen):
        super(My_Sequential, self).__init__()
        self.forecast_seqlen = forecast_seqlen

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        forecast = torch.zeros(size=(batch_size, self.forecast_seqlen, 1))
        my_log.info_logger(f"stack.inputs:{inputs.shape}")
        for module in self:
            forecast_, input_estimate = module(inputs)
            forecast +=forecast_
            inputs = input_estimate

        my_log.info_logger(f"stack.output_forecast:{forecast.shape, input_estimate.shape}")
        return forecast, input_estimate

class stack(nn.Module):
    def __init__(self, input_size:int=2, 
                        hidden_size_list:Vector=[4,5,6,7,8], 
                        input_seqlen:int=14, 
                        forecast_seqlen:int=3):
        super(stack,self).__init__()
        
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
            forecast : 所有basicblock的预测输出之和
            input_estimate: 模型最后的输出，也就是输入 减  最佳估计
        '''
        
        # forcast value ，之后的每一个block输出的预测值都加进来
        batch_size = inputs.shape[0]
        forecast = torch.zeros(size=(batch_size, self.forecast_seqlen, 1))
        my_log.info_logger(f"stack.intputs: {inputs.shape}")
        my_log.info_logger(f"stack.output_forecast: {forecast.shape}")
        
        
        # forecast_ 和 x_ 都是临时变量 暂时存放当前 block 的输出
        forecast_, x_ = self.block_1(inputs)
        forecast +=forecast_
        inputs = x_
        my_log.info_logger(f"stack.block_1_output :{forecast_.shape, x_.shape}")
    


        forecast_, x_ = self.block_2(inputs)
        forecast +=forecast_
        inputs = x_
        my_log.info_logger(f"stack.block_2_output :{forecast_.shape, x_.shape}")
        
        '''
        # 使用Sequential方法
        forecast, input_estimate = self.net(inputs)
        '''
        return forecast, inputs




if __name__ == '__main__':
    
    batch_size = 128
    seq_len = 14
    forecast_seqlen = 3
    features = 2
    x = torch.rand(size=(batch_size, seq_len, features))
    # model = BasicBlock(input_size=features, hidden_size=6, estimate_seqlen=seq_len, forecast_seqlen=forecast_seqlen)
    model = stack(input_size=features, hidden_size_list=[6,5], input_seqlen=seq_len, forecast_seqlen=forecast_seqlen)
    forecast, input_estimate = model(x)
    
    

    
    