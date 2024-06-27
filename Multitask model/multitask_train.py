from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import torch
from zincdataset import *
from multitaskmodel import MultiTaskModel
from torch import nn
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('load dataset')
sdf_file_train ='Zinc_dataset/zinc_250k_std_train.sdf' 
sdf_file_test ='Zinc_dataset/zinc_250k_std_validation.sdf' 

train_dataset = MoleculeDataset(sdf_file_train)
test_dataset = MoleculeDataset(sdf_file_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = MultiTaskModel(num_node_features=9, hidden_channels=128,  num_layers = 3,num_classes=9, out_channels=3).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
criterion_classification = nn.CrossEntropyLoss()
criterion_regression = nn.MSELoss()

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds_classification = []
    all_labels_classification = []
    all_preds_regression = []
    all_labels_regression = []
    for data in loader:
        data = mask_node_labels(data)
        data = data.to(device)
        optimizer.zero_grad()
        out_classification, out_regression = model(data.x, data.edge_index, data.batch)
        #print(out_classification.shape, out_regression.shape, data.x.shape)
        loss_classification = criterion_classification(out_classification[data.mask], data.mapped_labels)
        loss_regression = criterion_regression(out_regression, data.y)
        
        loss = loss_classification + loss_regression
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted_index = out_classification[data.mask].max(dim=1)
            predicted = [index_to_atom_type[p.item()] for p in predicted_index]
            #predicted = torch.tensor(predicted)
            all_preds_classification.extend(predicted)
            all_labels_classification.extend(data.masked_node_labels.cpu().numpy())
            
            all_preds_regression.extend(out_regression.cpu().numpy())
            all_labels_regression.extend(data.y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels_classification, all_preds_classification)
    mse = mean_squared_error(all_labels_regression, all_preds_regression)
    r2 = r2_score(all_labels_regression, all_preds_regression)
    
    return total_loss / len(loader), accuracy, mse, r2


def test(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds_classification = []
    all_labels_classification = []
    all_preds_regression = []
    all_labels_regression = []
    with torch.no_grad():
        for data in loader:
            data = mask_node_labels(data)
            data = data.to(device)
            out_classification, out_regression = model(data.x, data.edge_index, data.batch)
            
            loss_classification = criterion_classification(out_classification[data.mask], data.mapped_labels)
            loss_regression = criterion_regression(out_regression, data.y)
            
            loss = loss_classification + loss_regression
            total_loss += loss.item()
            
            _, predicted_index = out_classification[data.mask].max(dim=1)
            predicted = [index_to_atom_type[p.item()] for p in predicted_index]
            #predicted = torch.tensor(predicted)
            all_preds_classification.extend(predicted)
            all_labels_classification.extend(data.masked_node_labels.cpu().numpy())
            
            all_preds_regression.extend(out_regression.cpu().numpy())
            all_labels_regression.extend(data.y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels_classification, all_preds_classification)
    mse = mean_squared_error(all_labels_regression, all_preds_regression)
    r2 = r2_score(all_labels_regression, all_preds_regression)
    
    return total_loss / len(loader), accuracy, mse, r2

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
train_mses = []
test_mses = []
train_r2s = []
test_r2s = []

for epoch in range(1, 201):
    train_loss, train_acc, train_mse, train_r2 = train(model, train_loader, optimizer, device)
    test_loss, test_acc, test_mse, test_r2 = test(model, test_loader, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    train_mses.append(train_mse)
    test_mses.append(test_mse)
    train_r2s.append(train_r2)
    test_r2s.append(test_r2)

    if epoch %10 == 0:
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}')

path = 'zinc_multitask'
import os 
import pickle
if not os.path.exists(path):
    os.makedirs(path)

torch.save(model.state_dict(), f'{path}/model.pth')
with open(f'{path}/train_losses.txt', 'wb') as file:
       pickle.dump(train_losses, file)

with open(f'{path}/test_losses.txt', 'wb') as file:
       pickle.dump(test_losses, file)

with open(f'{path}/train_accuracies.txt', 'wb') as file:
       pickle.dump(train_accuracies, file)

with open(f'{path}/test_accuracies.txt', 'wb') as file:
       pickle.dump(test_accuracies, file)

with open(f'{path}/train_mses.txt', 'wb') as file:
       pickle.dump(train_mses, file)

with open(f'{path}/test_mses.txt', 'wb') as file:
       pickle.dump(test_mses, file)

with open(f'{path}/train_r2s.txt', 'wb') as file:
       pickle.dump(train_r2s, file)

with open(f'{path}/test_r2s.txt', 'wb') as file:
       pickle.dump(test_r2s, file)