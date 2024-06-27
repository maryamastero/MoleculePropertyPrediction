import torch
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, GINConv
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

unique_atom_types = [35, 6, 7, 8, 9, 15, 16, 17, 53]
atom_type_to_index = {int(atom_type): index for index, atom_type in enumerate(unique_atom_types)}

index_to_atom_type = {index: int(atom_type) for index, atom_type in enumerate(unique_atom_types)}

def mask_node_labels(data, mask_ratio=0.2, mask_value=-1):
    num_nodes = data.num_nodes
    mask = torch.rand(num_nodes) < mask_ratio
    data.mask = mask

    data.masked_node_labels = data.z[mask].clone()
    data.masked_node_features = data.x[mask].clone()

    data.z[mask] = mask_value
    data.x[mask] = mask_value
    mapped_labels = [atom_type_to_index[z.item()] for z in data.masked_node_labels]
    mapped_labels = torch.tensor(mapped_labels)
    data.mapped_labels = mapped_labels
    return data

import matplotlib.pyplot as plt

def acc_loss_plots(train_losses, train_accs, valid_losses, valid_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(valid_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(valid_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()




if __name__ =='__main__':
    from pretrain_model import *
    from zincdataset import *

    sdf_file = 'Zinc_dataset/zinc_250k_std_test.sdf'
    dataset = MoleculeDataset(sdf_file)
            
    train_dataset = dataset[:110]
    batch_size = 1
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = next(iter(train_loader))
    data = batch.to(device)
    data = mask_node_labels(data)
    print('masked labels', data.masked_node_labels)
    
    model = MaskedAtomIdentification(data.x.shape[1], 64, 3,9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    out = model(data.x, data.edge_index)
    print('out[data.mask]', out[data.mask])
    print(out.shape)
    batch_size = 32
    loss = criterion(out[data.mask], data.masked_node_labels)
    print(loss)

