from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data, Dataset
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def read_molecules_from_sdf(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    molecules = [mol for mol in supplier if mol is not None]
    return molecules

def get_property_values(mol, label_property_names):
    property_values = []
    for prop in label_property_names:
        if mol.HasProp(prop):
            prop_value = mol.GetProp(prop).split('\n')
            property_values.append(prop_value)
    return property_values

def parse_properties_list(properties_list):
    property_dict = {}
    current_key = None
    property_name_mapping = {
        'logP': 'LogP',
        'QED': 'QED',
        'SAS': 'SAS',
        'SMILES': 'SMILES'
    }

    for item in properties_list[0]:
        if item.startswith('>  <') and item.endswith('>'):
            raw_key = item[4:-1]
            current_key = property_name_mapping.get(raw_key, raw_key)
        elif current_key:
            if current_key in ['LogP', 'QED', 'SAS']:
                property_dict[current_key] = float(item)
            else:
                property_dict[current_key] = item
            current_key = None
        elif current_key is None and item.strip() and not item.startswith('$$$$'):
            property_dict['LogP'] = float(item)

    return property_dict

e_map = {'bond_type': ['misc', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
        'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'],
        'is_conjugated': [False, True],}

x_map = {
    'atomic_num': list(range(0, 119)),
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': ['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],}

def mol_to_data(mol, label_property_names):
    # Atom features is just atomic number
    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)
    
    x = torch.tensor(xs, dtype=torch.float).view(-1, 9)

    # Edge index and edge attributes
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]
    
    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    # Get property values and ensure y has a consistent shape
    property_values = get_property_values(mol, label_property_names)
    property_dict = parse_properties_list(property_values)
    y_values = [property_dict.get(prop, 0.0) for prop in ['LogP', 'QED', 'SAS']]  # Default to 0.0 if property is missing
    y = torch.tensor(y_values, dtype=torch.float).view(-1, 3)  # Ensure y is a tensor of shape [num_properties]

    z = [x_map['atomic_num'].index(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    z = torch.tensor(z, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, z=z, smiles=Chem.MolToSmiles(mol))

class MoleculeDataset(Dataset):
    def __init__(self, sdf_file, transform=None, pre_transform=None):
        self.sdf_file = sdf_file
        self.label_property_names = ['logP', 'QED', 'SAS', 'smiles']
        super(MoleculeDataset, self).__init__('.', transform, pre_transform)
        self.molecules = read_molecules_from_sdf(sdf_file)

    def len(self):
        return len(self.molecules)

    def get(self, idx):
        mol = self.molecules[idx]
        data = mol_to_data(mol, self.label_property_names)
        return data

if __name__ =='__main__':
    sdf_file_test ='Zinc_dataset/zinc_250k_std_test.sdf' 

    train_dataset = MoleculeDataset(sdf_file_test)
    test_dataset = MoleculeDataset(sdf_file_test)
    # Create DataLoaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset[:5], batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset[:5], batch_size=64, shuffle=False)
    batch = next(iter(train_loader))
    print(batch.y)



