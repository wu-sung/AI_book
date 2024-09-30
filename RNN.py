import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

path = "C:/Users/chosh/OneDrive/바탕 화면/pycharm/Sung/"

df = pd.read_csv(path + 'kospi.csv')
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print(f'Using device: {device}')

#종가 제외 스케일링
scaler = MinMaxScaler()
df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open','High', 'Low','Close','Volume']])
x = df[['Open', 'High', 'Low', 'Volume']].values
y = df['Close'].values
def sequence_data(x, y, sequence_length):
    x_seq = []
    y_seq = []
    for i in range(len(x) - sequence_length):
        x_seq.append(x[i: i + sequence_length])
        y_seq.append(y[i + sequence_length])
        # gpu용 텐서로 변환
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view(-1, 1)

split = 200
sequence_length = 5
x_seq, y_seq = sequence_data(x, y, sequence_length)
# 순서대로 200개는 학습, 나머지는 평가
x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]

print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size)

#기본 하이퍼 파라미터 설정
input_size = x_seq.size(2)
num_layers = 2
hidden_size = 8

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(VanillaRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # 한 줄로 모델 정의
        self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1),
                                nn.Sigmoid())  # RNN 층에서 나온 결과를 fc 층으로 전달해서 예측값 계산

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)  # 초기값 0으로 설정
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

model = VanillaRNN(input_size, hidden_size, sequence_length, num_layers, device).to(device)

criterion = nn.MSELoss()
num_epochs = 301
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_graph = []
n = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0

    for data in train_loader:
        seq, target = data  # 배치 데이터
        out = model(seq)  # 출력값 계산
        loss = criterion(out, target)  # 손실함수 계산

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 최적화
        running_loss += loss.item()

    loss_graph.append(running_loss / n)
    print('[epoch : %d] loss: %.4f' % (epoch, running_loss / n))


concatdata = torch.utils.data.ConcatDataset([train, test])
data_loader = torch.utils.data.DataLoader(dataset=concatdata, batch_size=100)

with torch.no_grad():
    pred = []
    model.eval()
    for data in data_loader:
        seq, target = data
        out = model(seq)
        pred += out.cpu().tolist()

plt.figure(figsize = (20,10))
plt.plot(np.ones(100)*len(train), np.linspace(0,1,100), '--', linewidth=0.6)
plt.plot(df['Close'][sequence_length:].values,'--')
plt.plot(pred, 'b', linewidth = 0.6)
plt.legend(['train_boundary','actual','prediction'])
plt.show()
