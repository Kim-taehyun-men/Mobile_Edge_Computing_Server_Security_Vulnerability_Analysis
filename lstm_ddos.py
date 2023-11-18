import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

packet_data = pd.read_csv('labeled_packets.csv')

packet_data = pd.get_dummies(packet_data, columns = ['Source','Destination', 'Protocol', 'Info'])

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        if len(_x.loc[_x['Label'] == 1]) > 9:
            _y = 1        
        else:
            _y = 0
        x.append(_x.drop(columns=['Label']))
        y.append([_y])


    return np.array(x),np.array(y)

seq_length = 10


x_packet, y_packet = sliding_windows(packet_data, seq_length)

x_packet.shape, y_packet.shape


train_size = int(len(x_packet)*0.7)
train_x = x_packet[:train_size]

valid_x = x_packet[train_size:]

train_size = int(len(x_packet)*0.7)
train_y = y_packet[:train_size]

valid_y = y_packet[train_size:]

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
  def __init__(self, data_x, data_y):
    self.data_x = data_x
    self.data_y = data_y

  def __len__(self):
    return self.data_y.shape[0]

  def __getitem__(self, idx):
    x = torch.from_numpy(self.data_x[idx])
    y = torch.from_numpy(self.data_y[idx])
    return x,y

  def get_labels(self):
    return list(train_y.squeeze())


n_hidden = 128

n_features = 12435

n_labels = 2

n_layer = 3
batch_size = 128

num_epochs = 1000

learning_rate = 0.0001


train_dataset = CustomDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size = batch_size,
                                          shuffle=True)

test_dataset = CustomDataset(valid_x, valid_y)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = batch_size,
                                          shuffle=False)

class LSTM(nn.Module):
       #그냥 실행
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim,output_dim)
        self.bn = nn.BatchNorm1d(seq_length)

    def forward(self,inputs):
        x = self.bn(inputs)
        lstm_out,(hn,cn) = self.lstm(x)
        out = self.fc(lstm_out[:,-1,:])
        return out

model = LSTM(n_features,n_hidden,n_labels,n_layer)

model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

all_losses = []
current_loss = 0
_loss = 100

total_step = len(train_loader)
for epoch in range(num_epochs):
   for i, data in enumerate(train_loader):
      inputs, labels = data
      inputs = inputs.to(device).float()
      labels = labels.to(device).long()
      labels = torch.squeeze(labels)

      outputs = model(inputs)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      current_loss += loss.item()

   if (epoch+1) % 10 == 0:
      print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
      all_losses.append(current_loss/1000)
      current_loss = 0
      if _loss > loss:
         _loss = loss
         torch.save(model.state_dict(),f'./lstm_{epoch+1}.pt')


torch.save(model.state_dict(),f'./lstm_last.pt')


model = LSTM(n_features,n_hidden,n_labels,n_layer)

model.load_state_dict(torch.load('./lstm_20.pt'))

model.to(device)


from sklearn.metrics import f1_score
#테스트 하는것
model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    y_pred = np.array([])  
    y_valid = np.array([]) 


    for inputs, labels in test_loader:        
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        labels = torch.squeeze(labels)
        outputs = model(inputs)               
        _, predicted = torch.max(outputs, 1)
        y_pred = np.concatenate([y_pred, np.array(predicted.cpu())]) 
        y_valid = np.concatenate([y_valid, np.array(labels.cpu())])  

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    print('Accuracy          :  {} %'.format(100 * correct / total))

  
    macro = f1_score(y_valid, y_pred, average='macro')
    weighted = f1_score(y_valid, y_pred, average='weighted')
    micro = f1_score(y_valid, y_pred, average='micro')
    f1 = f1_score(y_valid, y_pred)
    print('f1-score          : ', f1*100, "%")
    print('f1-score macro    : ', macro*100, "%")
    print('f1-score weighted : ', weighted*100, "%")
    print('f1-score micro    : ', micro*100, "%")
