import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('current cuda device is', device)

batch_size = 50
learning_rate = 0.0001
epoch_num = 15

train_data = datasets.MNIST(root = './data',
                            train = True,
                            download= True,
                            transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data',
                           train = False,
                           transform = transforms.ToTensor())

print('number of traing data:',len(train_data))
print('number of test data:', len(test_data))
image, label = train_data[0]

plt.imshow(image.squeeze().numpy(), cmap ='gray')
plt.title('label: %s' % label)
plt.show()

train_loader = (torch.utils.data.DataLoader
                (dataset = train_data,
                  batch_size = batch_size, shuffle = True))
test_loader = (torch.utils.data.DataLoader
               (dataset = test_data,
                  batch_size = batch_size, shuffle = True))

first_batch = train_loader.__iter__().__next__()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(3136, 1000)  # 7*7*64=3436
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterison = nn.CrossEntropyLoss()

model.train()
i = 1
for epoch in range(epoch_num):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterison(output, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print('train step: {}\t loss: {:.3f}'.format(i, loss.item()))

        i += 1
model.eval()
correct = 0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('test set : accuracy:{:.2f}%'.format(100. * correct / len(test_loader.dataset)))