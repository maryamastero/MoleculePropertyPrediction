from utils import *
import pickle
from pretrain import MaskedAtomIdentification
import numpy as np
from zincdataset import *


path = 'zinc_06_13_128'



with open(f'{path}/train_losses.txt', "rb") as file:
        train_losses = pickle.load(file)

with open(f'{path}/train_accs.txt', "rb") as file:
       train_accs = pickle.load(file)
       
with open(f'{path}/valid_losses.txt', "rb") as file:
        valid_losses = pickle.load(file)

with open(f'{path}/valid_accuracies.txt', "rb") as file:
        valid_accuracies = pickle.load(file)

with open(f'{path}/predicted.txt', "rb") as file:
        predicted = pickle.load(file)

with open(f'{path}/actual.txt', "rb") as file:
        actual = pickle.load(file)

acc_loss_plots(train_losses, train_accs, valid_losses, valid_accuracies)

print("Predictions:", predicted[1])
print("Actual labels:", actual[1])
print('mean accuracy: ', np.mean(valid_accuracies))


sdf_file_test ='Zinc_dataset/zinc_250k_std_test.sdf' 

test_dataset =  MoleculeDataset(sdf_file_test)
# Create DataLoaders
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch = next(iter(test_loader))
data = batch.to(device)
data = mask_node_labels(data)


model = MaskedAtomIdentification(data.x.shape[1], 128, 3,9).to(device)
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(f'{path}/model.pth',map_location=torch.device('cpu')))
