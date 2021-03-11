import torch

blocks = []

class My_Sequential(nn.Sequential):
    def __init__(self, forecast_seqlen):
        super(My_Sequential,self).__init__()
        self.forecast_seqlen = forecast_seqlen

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        forecast = torch.zeros(size=(batch_size, self.forecast_seqlen, 1))
        for module in self:
            forecast_, inputs_ = module(inputs)
            forecast +=forecast_
            inputs = inputs_

        return forecast, inputs


for i in range(3):
    blocks += [s+str(i) = torch.nn.Linear(3,3)]

print(blocks)

