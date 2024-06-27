import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from pretrain_model import MaskedAtomIdentification
from utils import *
import matplotlib.pyplot as plt
from zincdataset import *
import os
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pickle
from torch import nn

from sklearn.metrics import mean_squared_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from finetune_model import PretrainedGINForPropertyPrediction






def train_property_prediction(model, loader, optimizer, criterion, device):
       model.train()
       total_loss = 0
       predictions = []
       true_values = []
       step = 0
       for data in loader:
              data = data.to(device)
              optimizer.zero_grad()
              out = model(data.x, data.edge_index, data.batch)
              loss = criterion(out, data.y)
              loss.backward()
              optimizer.step()
              total_loss += loss.item()
              predictions.append(out.cpu())
              true_values.append(data.y.cpu())

       predictions , true_values = torch.cat(predictions, dim=0), torch.cat(true_values, dim=0)

       predictions , true_values = predictions.detach().numpy()  , true_values.detach().numpy() 
       mse = mean_squared_error(true_values, predictions)
       r2 = r2_score(true_values, predictions)
       return total_loss / len(loader) , mse, r2

def test_property_prediction(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_values = []
    with torch.no_grad():
       for data in loader:
              data = data.to(device)
              out = model(data.x, data.edge_index, data.batch)
              loss = criterion(out, data.y)
              total_loss += loss.item()
              predictions.append(out.cpu())
              true_values.append(data.y.cpu())

       predictions , true_values = torch.cat(predictions, dim=0), torch.cat(true_values, dim=0)
       predictions , true_values = predictions.detach().numpy()  , true_values.detach().numpy() 
       mse = mean_squared_error(true_values, predictions)
       r2 = r2_score(true_values, predictions)
       return total_loss / len(loader) , mse, r2




print('load dataset')
sdf_file_train ='Zinc_dataset/zinc_250k_std_training.sdf' 
sdf_file_test ='Zinc_dataset/zinc_250k_std_validation.sdf' 

train_dataset = MoleculeDataset(sdf_file_train)
test_dataset = MoleculeDataset(sdf_file_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

batch = next(iter(train_loader))
data = batch.to(device)
data = mask_node_labels(data)

pretrained_model = MaskedAtomIdentification(data.x.shape[1], 128, 3,9).to(device)
pretrained_model.load_state_dict(torch.load('zinc_06_13_128/model.pth'))
pretrained_gin = pretrained_model.gin

property_prediction_model = PretrainedGINForPropertyPrediction(pretrained_gin, 128, 3).to(device)

optimizer = torch.optim.Adam(property_prediction_model.parameters(), lr=0.001)
criterion = nn.MSELoss()  


train_losses = []
train_mses = []
train_r2s = []

valid_losses = []
valid_r2s = []
valid_mses = []

for epoch in range(1, 201):
    train_loss, train_mse, train_r2 = train_property_prediction(property_prediction_model, train_loader, optimizer, criterion, device)
    valid_loss, valid_mse, valid_r2 = test_property_prediction(property_prediction_model, test_loader, criterion, device)

    train_losses.append(train_loss)
    train_mses.append(train_mse)
    train_r2s.append(train_r2)

    valid_losses.append(valid_loss)
    valid_mses.append(valid_mse)
    valid_r2s.append(valid_r2)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d} *****, Train Loss: {train_loss:.4f},\t Train MSE: {train_mse:.4f},\t Train R2: {train_r2:.4f},\t Test Loss: {valid_loss:.4f},\t Test MSE: {valid_mse:.4f},\t Test R2: {valid_r2:.4f}')

path = 'property_prediction_model/zinc_06_26'

if not os.path.exists(path):
    os.makedirs(path)

torch.save(property_prediction_model.state_dict(), f'{path}/property_prediction_model.pth')


with open(f'{path}/train_losses.txt', 'wb') as file:
       pickle.dump(train_losses, file)

with open(f'{path}/train_mses.txt', 'wb') as file:
       pickle.dump(train_mses, file)
       
with open(f'{path}/train_r2s.txt', 'wb') as file:
       pickle.dump(train_r2s, file)

with open(f'{path}/valid_losses.txt', 'wb') as file:
       pickle.dump(valid_losses, file)

with open(f'{path}/valid_mses.txt', 'wb') as file:
       pickle.dump(valid_mses, file)

with open(f'{path}/valid_r2s.txt', 'wb') as file:
       pickle.dump(valid_r2s, file)

