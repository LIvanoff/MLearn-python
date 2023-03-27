import pandas as pd
import torch
import torch.nn.functional as F


class NeuronBayes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 10)
        self.activ1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 3)
        self.activ2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(3, 2)
        self.activ3 = torch.nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ2(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        x = self.activ3(x)
        return x

    def predict(self, x):
        y_pred = self.forward(x)
        print(y_pred)
        pred = torch.argmax(y_pred, -1)
        labels = {0: 'Да', 1: 'Нет'}
        for label in labels.keys():
            if pred == label:
                return labels[label]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_excel('../bayes_test.xlsx')
data = train.values
y = torch.LongTensor(data[:, 3:4])
y = torch.transpose(y, 0, 1)
y = torch.LongTensor(y[0,:])
print(y.shape)
y = y.to(device)

x_train = torch.Tensor(data[:, :3])
x_train = x_train.to(device)
model = NeuronBayes()
model.train()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1.0e-4)

for epoch in range(100):
    for i in range(0, len(x_train)):
        pred = model.forward(x_train)
        loss = F.cross_entropy(pred, y)
        print('epoch: ' + str(epoch) + ' loss: ' + str(float(loss)))
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        test_preds = model.forward(x_train)
        test_preds = test_preds.argmax(dim=1)
        print((test_preds == y).float().mean())

model.eval()
y_test = torch.Tensor([1., 1., 0.])
print(model.predict(y_test))
