import os
import torch
import sys
import time
sys.path.append("..")
import seq2seq
from utils.my_logging import My_Logging as My_Logging
import utils.prepare_data as prepare_data
from alive_progress import alive_bar



DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
torch.set_default_tensor_type('torch.DoubleTensor')

lr = 1e-3
batch_size = 128

# 配置Logging
my_log = My_Logging()

# 准备数据
train_ip, train_op, test_ip, test_op = prepare_data.prepare_data(time_steps=14, horizion=3, features=2)
train_dataset = prepare_data.My_Train_Datasets(train_ip, train_op)
test_dataset = prepare_data.My_Test_Dataset(test_ip, test_op)

train_generator = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_generator = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def save_state(state,filename="my_state.pt"):
    print("Saving model and optimizer state")
    torch.save(state, filename)

def load_state(filename="my_state.pt"):
    print("Loading model and optimizer state")
    model.load_state_dict(torch.load(filename)['model'])
    optimizer.load_state_dict(torch.load(filename)['optimizer'])

def epoch_time(start_time, end_time):
    '''
    function: calculate the time of every epoch
    params:
        start_time: 
        end_time
    return:
        elapsed_mins:
        elapsed_secs:
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, loss_func):
    '''
    params:
        model: 训练的模型
        iterator：数据生成器
        optimizer：优化器
        loss_func：损失函数
    returns:
        每个样本的平均损失值
    '''
    model.train()
    epoch_loss = 0
    
    for i, (x,y) in enumerate(iterator):
        x = x.to(DEVICE)  # [batch_size, seq_len, feature]
        y = y.to(DEVICE)  # [batch_size, predict_seqlen]

        optimizer.zero_grad()
        # 输入到模型的是[batch_size, seq_len, features],模型输出是[batch_size,predict_seqlen, 1]
        output = model(x)

        loss = loss_func(output.squeeze(dim=2), y) # 将output [batch_size, predict_seqlen]
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    # len(iterator) 是 批次数量  即 样本总数/batch_size
    return epoch_loss / len(iterator)

def evalute(model, iterator, loss_func):
    '''
    模型评估函数
    params:
        model: 待评估模型
        iterator: 验证集生成器
        loss_func: 损失函数
    return: 损失均值
    '''
    model.eval()
    epoch_loss = 0
    for i,(x,y) in enumerate(iterator):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        output = model(x) # output [batch_size, predict_seqlen, 1]
        loss = loss_func(output.squeeze(dim=2), y)
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def train_model(model, train_generator, test_generator, optimizer, loss_func):       
    N_EPOCHS = 50
    best_valid_loss = float('inf')

    with alive_bar(total=N_EPOCHS, title='training') as bar:

        for epoch in range(N_EPOCHS):
            # 记录开始时间
            start_time = time.time()
            # 训练评估
            train_loss = train(model, train_generator, optimizer, loss_func)
            valid_loss  =evalute(model, test_generator, loss_func)
            # 记录结束时间
            end_time = time.time()
            # 计算 分钟、秒
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # 保存最好的模型
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                my_state = {'model':model.state_dict(), "optimizer":optimizer.state_dict()}
                save_state(state=my_state)

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.4f}')
            print(f'\t Val. Loss: {valid_loss:.4f}')

            # 更新进度条
            bar()

def main():
    
    load_model = False
    # 模型 优化器 损失函数
    model = seq2seq.RNN_Seq2Seq(input_size=2, hidden_size=5, predict_seqlen=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_func = torch.nn.MSELoss(reduction='mean')

    if load_model:
        load_state()  # 加载之前训练好的模型
    else:
        train_model(model, train_generator, test_generator, optimizer, loss_func) # 训练模型

# 未完待续。。。。

class Train():
    def __init__(self):
        self.N_EPOCHS = 50
    

    def save_state(state,filename="my_state.pt"):
        print("Saving model and optimizer state")
        torch.save(state, filename)

    def load_state(filename="my_state.pt"):
        print("Loading model and optimizer state")
        model.load_state_dict(torch.load(filename)['model'])
        optimizer.load_state_dict(torch.load(filename)['optimizer'])

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(501)
    main()