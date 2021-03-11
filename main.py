import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import logging
import argparse
import torch.optim as optim
from TCN import TemporalConvNet
from utils import data_generator

logging.basicConfig(filename='log.log',        
                    filemode='w',              
                    level=logging.INFO,         
                    format="%(asctime)s--%(lineno)d-%(message)s",   
					datefmt= "%Y-%m-%d  %H:%M:%S %a")  
def set_parser():
    parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='dropout applied to layers (default: 0.05)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit (default: 2)')
    parser.add_argument('--ksize', type=int, default=7,
                        help='kernel size (default: 7)')
    parser.add_argument('--levels', type=int, default=8,
                        help='# of levels (default: 8)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval (default: 100')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='initial learning rate (default: 2e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--nhid', type=int, default=25,
                        help='number of hidden units per layer (default: 25)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')
    return parser
args = set_parser().parse_args()
torch.manual_seed(args.seed)

class Model(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        '''
        input_size: 输入特征数
        output_size: 最后线性层输出的特征数
        num_channels: TCN的 各个stack的 特征数
        kerne_size ： 卷积核的大小

        维度变化为 [input_size=1, *num_channels, output_size=10
        '''
        super(Model, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (batch_size=128, in_channels=1, seq_len=784)"""
        # y1 : [batch_size=128, channels=25, seq_len=784] 
        y1 = self.tcn(inputs)
        # o [batch_size=128, channels=10]
        o = self.linear(y1[:, :, -1])
        # return F.log_softmax(o, dim=1)
        return o

root = './data/'    
batch_size = args.batch_size   # 128
n_classes = 10    # 输出分类为10
input_channels = 1 # 输入通道数为 1 
seq_length = int(784 / input_channels) # 序列长度 28*28
epochs = args.epochs # 20
steps = 0

print(args)
train_loader, test_loader = data_generator(root, batch_size)

# 设置每个stack 里面 卷积的特征数  默认为[25, 25, 25, 25, 25, 25, 25, 25]
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize # 7
lr = args.lr # 2e-3


model = Model(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout).cuda()

optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

criteriol = nn.CrossEntropyLoss()


def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data [batch_size=128, channels=1, seq_len=784]
        # target [batch_size]
        data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)  # 也可以把图像数据看作 channel=2, seq_len=784/2 的序列
        

        optimizer.zero_grad()
        output = model(data)  
        # loss = F.nll_loss(output, target)
        loss = criteriol(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            train_loss = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target)in enumerate(test_loader):
            
            data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            
            output = model(data)
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += criteriol(output, target)
            pred = output.data.max(1)[1]
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            correct += pred.eq(target).sum().item()

        test_loss /= (batch_idx + 1)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr