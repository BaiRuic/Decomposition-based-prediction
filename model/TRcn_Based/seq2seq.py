import torch
import torch.nn as nn
from utils import my_logging
from torch.nn.utils import weight_norm
from . encoder_tcn import TcnEncoder

# 配置日志
my_log = my_logging.My_Logging()
my_log.info_logger("we")


# 解码器
class GruDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(GruDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size=self.input_size, 
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=True)

    def forward(self, inputs, prev_hidden):
        '''
        params:
            inputs: [batch_size, features=1, seq_len=1]
            prev_hidden [1, batch_size, hidden_size]
        returns:
            prediction: [batch_size, feature=1， seq_len=1]
            hidden: [1, batch_size, hidden_size]
        '''
        # output [batch_size, features=hidden_size, seq_len=1]
        # hidden [1, batch_size, hidden_size]
        output, hidden =  self.gru(inputs, prev_hidden)
        # prediction [batch_size, output_size=1, seq_len=1]
        prediction = self.fc(output.squeeze(dim=1)).unsqueeze(dim=1)
        return prediction, hidden


class BasicBlock(nn.Module):
    def __init__(self,input_size, hidden_size, forecast_seqlen, estimate_seqlen, kernel_size=3):
        '''
        input_size: 输出序列数据的样本特征
        hidden_size: 隐藏层样本特征
        predice_seqlen: 预测的时间步
        estimate_seqlen: 估计的时间步，即输入样本的时间步
        '''
        super(BasicBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.forecast_seqlen = forecast_seqlen
        self.estimate_seqlen = estimate_seqlen
        
        self.encoder = TcnEncoder(num_inputs=self.input_size, num_channels=[self.hidden_size], kernel_size=self.kernel_size)
        self.f_decoder = GruDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)
        self.e_decoder = GruDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, features]
        '''
        # from [batch_size, seq_len, features]  to   [batch_size, features, seq_len] 
        inputs = inputs.permute(0,2,1)


        batch_size = inputs.shape[0]

        # 对输入进行编码 得到编码状态 
        hidden = self.encoder(inputs)
        hidden.contiguous()
        my_log.info_logger(f"Encoder_input:  inputs:{inputs.shape}")
        my_log.info_logger(f"Encoder_output: hidden:{hidden.shape}")
        
        assert(hidden.shape==(1, batch_size, self.hidden_size))
      

        # 解码器输入
        f_decoder_input = inputs[:, 0, -1].reshape(batch_size, 1, 1)  # for forecast decoder
        e_decoder_input = inputs[:, 0, 0].reshape(batch_size, 1, 1)  # for estimate decoder   这里感觉时间步变成第一步比较好

        my_log.info_logger(f"f_Dncoder_input:  f_decoder_input:{f_decoder_input.shape}")
        my_log.info_logger(f"e_Dncoder_input:  e_decoder_input:{e_decoder_input.shape}")

        assert(f_decoder_input.shape==(batch_size, 1, 1))
        assert(e_decoder_input.shape==(batch_size, 1, 1))

        # 存放 两个解码器的输出列表
        forecast_outputs = []  
        estimate_outputs = []  

        # 初始化 解码器状态
        f_hidden = hidden   # for forecast decoder
        e_hidden = hidden   # for estimate decoder

        # 做预测
        for _ in range(self.forecast_seqlen):
            out, f_hidden = self.f_decoder(f_decoder_input, f_hidden)
            forecast_outputs.append(out)
            f_decoder_input = out

        # 做估计
        for _ in range(self.estimate_seqlen):
            out, e_hidden = self.e_decoder(e_decoder_input, e_hidden)
            estimate_outputs.append(out)    
            e_decoder_input = out
        
        # 将列表 estimate_outputs 和 forecast_outputs 转换为tensor
        estimate_outputs = torch.cat(estimate_outputs, dim=2)  # [batch_size, 1, seq_len=estimate_seqlen]
        forecast_outputs = torch.cat(forecast_outputs, dim=2)  # [batch_size, 1, seq_len=estimate_seqlen]

        # 提取输入数据中的 需要预测变量数据 如负载预测中，提取负载数据。剔除温度数据
        input_main = inputs[:,0,:].unsqueeze(dim=1) # [batch_size, features=1, seq_len]
        my_log.info_logger(f"BasicBlock_output:input_main:{input_main.shape}")
        my_log.info_logger(f"BasicBlock_output:estimate_outputs:{estimate_outputs.shape}")
        my_log.info_logger(f"BasicBlock_output:forecast_outputs:{forecast_outputs.shape}")
        assert(input_main.shape == estimate_outputs.shape)

        # from [batch_size, features=1, seq_len]  to  [batch_size, seq_len, features=1]
        forecast_outputs = forecast_outputs.permute(0, 2, 1)
        input_main = input_main.permute(0, 2, 1)
        estimate_outputs = estimate_outputs.permute(0, 2, 1)
        return forecast_outputs, input_main-estimate_outputs


if __name__ == "__main__":
    my_log.info_logger('begin')
    batch_size = 128
    time_step = 14
    features = 2
   # 测试输入 (batch_size, time_step, features) 是否可行
    x = torch.rand(size =(batch_size, time_step, features) )


    '''
    encoder = TcnEncoder(num_inputs=features, 
                            num_channels=[4,6,5],
                            kernel_size=3,
                            dropout=0.5)
    prev_hidden = encoder(x)

    my_log.info_logger(f"prev_hidden{prev_hidden.shape}")
    decoder = GruDecoder(input_size=1, hidden_size=5)

    decoder_input = x[:, 0, -1].reshape(batch_size, 1, 1)
    pred, hidden = decoder(decoder_input, prev_hidden)
    my_log.info_logger(f"{pred.shape}")
    my_log.info_logger(f"{hidden.shape}")
    '''
    model = BasicBlock(input_size=2, hidden_size=5, forecast_seqlen=7, estimate_seqlen=14)
    y, _ = model(x)
    print(y.shape)