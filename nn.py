
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim 
class Dataset_py(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]



y_train_array, y_test_array = y_train.values, y_test.values
y_train_array = y_train_array.reshape((len(y_train_array), 1))
y_test_array = y_test_array.reshape((len(y_test_array), 1))

batch_size = 32
train_ds = Dataset_py(x_train, y_train_array)
# test_ds = Dataset_py(x_test, y_test_array)
train_dl = DataLoader(train_ds, batch_size=batch_size)
xtrain_t = torch.tensor(x_train, dtype=torch.float32)
xtest_t = torch.tensor(x_test, dtype=torch.float32)

class MLP(nn.Module):
  def __init__(self, input_size, output_size):
    super(MLP, self).__init__()
    self.net = nn.Sequential(
    nn.Linear(input_size, 64), 
    nn.ReLU(), 
    nn.Linear(64, 32), 
    nn.Linear(32, output_size),
    )

  def forward(self, x):
    pred = self.net(x)
    return pred 


nn_model = MLP(x_train.shape[1], 1)
cost = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
loss_list = []

epoch_num = 30
for epoch in range(epoch_num):
  for x_batch, y_batch in train_dl:
      pred = nn_model(x_batch)
      loss = torch.sqrt(cost(pred, y_batch))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  print('epoch %s, error %s'%(epoch, loss.item()))
  loss_list.append(loss.item())

nn_model.eval()
nn_y = nn_model(xtest_t).detach().numpy()
nn_yt = nn_model(xtrain_t).detach().numpy()
nn_test_error = np.sqrt(mean_squared_error(nn_y, y_test_array))
nn_train_error = np.sqrt(mean_squared_error(nn_yt, y_train_array))
print('train_error, test_error', nn_train_error, nn_test_error)
train_error_list.append(nn_train_error)
test_error_list.append(nn_test_error)

