import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

binary_datas = pd.read_excel('binary_classification_data.xlsx')
datas = binary_datas.iloc[:, :2].values
labels = binary_datas.iloc[:, 2].values

plt.figure(figsize=(8, 6))
for data in binary_datas.values:
    if data[2] == 1:
        plt.scatter(data[0], data[1], color='blue')
    else:
        plt.scatter(data[0], data[1], color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Data Visualization")
plt.show()

input_train, input_test, label_train, label_test = train_test_split(datas, labels, test_size=0.1, stratify=labels)

input_train_tensor = torch.tensor(input_train, dtype=torch.float32)
label_train_tensor = torch.tensor(label_train, dtype=torch.float32).unsqueeze(1)
input_test_tensor = torch.tensor(input_test, dtype=torch.float32)
label_test_tensor = torch.tensor(label_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(input_train_tensor, label_train_tensor)
test_dataset = TensorDataset(input_test_tensor, label_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(2, 4)
        self.fc2 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

model.train()

for epoch in range(num_epochs):

    epoch_loss = 0.0
    correct = 0
    total = 0

    for input_batch, label_batch in train_loader:
        outputs = model(input_batch)
        loss = criterion(outputs, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        predicted = (outputs > 0.5).float()
        total += label_batch.size(0)
        correct += (predicted == label_batch).sum().item()

    epoch_loss /= len(train_loader)
    epoch_accuracy = correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

th = 0.5
plt.figure(figsize=(8, 6))
model.eval()

with torch.no_grad():
    result = np.empty([1000 * 2, 3], dtype=np.float32)
    correct = 0
    total = 0
    test_loss = 0.0

    for input_batch, label_batch in test_loader:
        outputs = model(input_batch)

        for i in range(len(outputs)):
            if (outputs[i] < th) and (label_batch[i] == 0):
                plt.scatter(input_batch[i, 0], input_batch[i, 1], color='blue')
            elif (outputs[i] >= th) and (label_batch[i] == 1):
                plt.scatter(input_batch[i, 0], input_batch[i, 1], color='red')
            elif (outputs[i] < th) and (label_batch[i] == 1):
                plt.scatter(input_batch[i, 0], input_batch[i, 1], color='yellow')
            elif (outputs[i] >= th) and (label_batch[i] == 0):
                plt.scatter(input_batch[i, 0], input_batch[i, 1], color='green')

        loss = criterion(outputs, label_batch)
        predicted = (outputs > th).bool()
        total += label_batch.size(0)
        correct += (predicted == label_batch.bool()).sum().item()
        test_loss += loss.item()

    accuracy = correct / total
    test_loss /= len(test_loader)

print(f'Final Loss: {test_loss:.4f}, Final Accuracy: {accuracy:.4f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('prediction')
plt.show()
