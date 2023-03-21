from torch import nn
from torch import optim
import pandas as pd
import torch
import torch.nn.functional as F


class Neuron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 1)
        self.activ1 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        return x

    def predict(self, x):
        return self.forward(x)


train = pd.read_excel('../bayes_test.xlsx')
data = train.values
y = torch.Tensor(data[:, 3:4])
x_train = torch.Tensor(data[:, :3])
model = Neuron()
model.train()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1.0e-4)

for epoch in range(1300):
    for i in range(0, len(x_train)):
        pred = model.forward(x_train)
        loss = F.binary_cross_entropy(pred, y)
        print('epoch: ' + str(epoch) + ' loss: ' + str(float(loss)))
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        test_preds = model.forward(x_train)
        test_preds = test_preds.argmax(dim=1)
        print((test_preds == y).float().mean())

model.eval()
y_test = torch.Tensor([1., 1., 1.])
print(model.predict(y_test))
