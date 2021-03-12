import os
import sys
sys.path.append("..")
import seq2seq
import utils.my_logging as my_logging
import utils.prepare_data as prepare_data


my_log = my_logging.My_Logging()

model = seq2seq.RNN_Seq2Seq(input_size=2, hidden_size=5)

train_ip, train_op, test_ip, test_op = prepare_data.prepare_data(time_steps=14, horizion=3, features=2)
train_dataset = prepare_data.My_Train_Datasets(train_ip, train_op)
test_dataset = prepare_data.My_Test_Dataset(test_ip, test_op)



for x, y in train_dataset:
    print(x.shape,y.shape)
    break

