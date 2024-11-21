import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils import data
from torch.optim import Adadelta
from sklearn.model_selection import train_test_split

from stgcn_models import *

# Configuration
save_folder = 'weights'
os.makedirs(save_folder, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100  # Reduced from 100000 for practicality
batch_size = 32
learning_rate = 0.001
time_steps = 30  # Fixed sequence length

dataset_path = 'pickle_folder'

# Get class names and number of classes
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
num_classes = len(class_names)
print('Name of Classes:', class_names)
print('Number of Classes:', num_classes)

# Map class names to indices
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# Gather all .pkl files from each class subdirectory
data_files = []
for class_name in class_names:
    class_dir = os.path.join(dataset_path, class_name)
    for file in os.listdir(class_dir):
        if file.endswith('.pkl'):
            data_files.append(os.path.join(class_dir, file))

print(f'Total .pkl files found: {len(data_files)}')


def load_dataset(data_files, class_to_idx, batch_size, split_size=0.2, fixed_T=30, num_classes=9):
    """Load data files into torch DataLoader with splitting into train and validation sets."""
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts = pickle.load(f)  # Load features
            fts = np.array(fts)[:, 0, :, :]  #
            
            # Handle fixed sequence length
            frame_num = fts.shape[0]
            if frame_num < fixed_T:
                pad_length = fixed_T - frame_num
                fts = np.pad(fts, ((0, pad_length), (0,0), (0,0)), 'constant')
            elif frame_num > fixed_T:
                start = np.random.randint(0, frame_num - fixed_T)
                fts = fts[start:start + fixed_T, :, :]
            
            features.append(fts)  # [fixed_T, 17, 3]
            
            # Assign label based on class directory
            class_name = os.path.basename(os.path.dirname(fil))
            class_idx = class_to_idx[class_name]
            label = np.zeros(num_classes)
            label[class_idx] = 1
            labels.append(label)
    
    features = np.array(features)  # [N, fixed_T, 17, 3]
    labels = np.array(labels)      # [N, num_classes]
    
    # Split into training and validation sets
    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(
            features, labels, test_size=split_size, random_state=9, stratify=labels
        )
        train_set = data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
            torch.tensor(y_train, dtype=torch.float32)
        )
        valid_set = data.TensorDataset(
            torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
            torch.tensor(y_valid, dtype=torch.float32)
        )
        train_loader = data.DataLoader(
            train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        valid_loader = data.DataLoader(
            valid_set, batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        train_set = data.TensorDataset(
            torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
            torch.tensor(labels, dtype=torch.float32)
        )
        train_loader = data.DataLoader(
            train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        valid_loader = None
    return train_loader, valid_loader


def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


def set_training(model, mode=True):
    model.train(mode)
    return model


if __name__ == '__main__':
    # DATA.
    train_loader, valid_loader = load_dataset(data_files, class_to_idx, batch_size, split_size=0.2, fixed_T=time_steps, num_classes=num_classes)
    dataloader = {'train': train_loader, 'valid': valid_loader}
    
    # MODEL.
    graph_args = {'strategy': 'spatial'}
    model = TwoStreamSpatialTemporalGraph(graph_args, num_classes).to(device)
    
    # Optimizer
    optimizer = Adadelta(model.parameters(), lr=learning_rate)
    
    # Loss Function
    losser = torch.nn.BCELoss()
    
    # TRAINING.
    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}
    for e in range(epochs):
        print(f'Epoch {e+1}/{epochs}')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model = set_training(model, True)
            else:
                model = set_training(model, False)
        
            run_loss = 0.0
            run_accu = 0.0
            if phase == 'valid' and valid_loader is None:
                continue
            with tqdm(dataloader[phase], desc=phase) as iterator:
                for pts, lbs in iterator:
                    # Create motion input by distance of points (x, y) of the same node
                    # in two frames.
                    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        
                    mot = mot.to(device)
                    pts = pts.to(device)
                    lbs = lbs.to(device)
        
                    # Forward.
                    out = model((pts, mot))
                    loss = losser(out, lbs)
        
                    if phase == 'train':
                        # Backward.
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
        
                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(),
                                          lbs.detach().cpu().numpy())
                    run_accu += accu
        
                    iterator.set_postfix_str(f' loss: {loss.item():.4f}, accu: {accu:.4f}')
        
            loss_list[phase].append(run_loss / len(iterator))
            accu_list[phase].append(run_accu / len(iterator))
        
        # Summary of the epoch
        if 'valid' in loss_list and len(loss_list['valid']) > 0:
            print(f'Summary epoch:\n - Train loss: {loss_list["train"][-1]:.4f}, accu: {accu_list["train"][-1]:.4f}\n - Valid loss: {loss_list["valid"][-1]:.4f}, accu: {accu_list["valid"][-1]:.4f}')
        else:
            print(f'Summary epoch:\n - Train loss: {loss_list["train"][-1]:.4f}, accu: {accu_list["train"][-1]:.4f}')
    
        # SAVE.
        torch.save(model.state_dict(), os.path.join(save_folder, 'tsstg-model.pth'))
    
    del train_loader, valid_loader
    
    # EVALUATION.
    model.load_state_dict(torch.load(os.path.join(save_folder, 'tsstg-model.pth')))
    model = set_training(model, False)
    eval_loader, _ = load_dataset(data_files, class_to_idx, batch_size, split_size=0.0, fixed_T=time_steps, num_classes=num_classes)
    
    print('Evaluation.')
    run_loss = 0.0
    run_accu = 0.0
    y_preds = []
    y_trues = []
    with tqdm(eval_loader, desc='eval') as iterator:
        for pts, lbs in iterator:
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot = mot.to(device)
            pts = pts.to(device)
            lbs = lbs.to(device)
    
            out = model((pts, mot))
            loss = losser(out, lbs)
    
            run_loss += loss.item()
            accu = accuracy_batch(out.detach().cpu().numpy(),
                                  lbs.detach().cpu().numpy())
            run_accu += accu
    
            y_preds.extend(out.argmax(1).detach().cpu().numpy())
            y_trues.extend(lbs.argmax(1).cpu().numpy())
    
            iterator.set_postfix_str(f' loss: {loss.item():.4f}, accu: {accu:.4f}')
    
    run_loss = run_loss / len(iterator)
    run_accu = run_accu / len(iterator)
    
    print(f'Eval Loss: {run_loss:.4f}, Accu: {run_accu:.4f}')
