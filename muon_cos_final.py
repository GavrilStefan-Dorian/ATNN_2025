import os
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
import numpy as np
import pickle
import time


device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
enable_half = device.type != "cpu"
scaler = GradScaler(device, enabled=enable_half)

print("Grad scaler is enabled:", enable_half)
print("Device:", device)

IN_COLAB = False
if IN_COLAB:
    print(f"Running on Google Colab - Using Google Drive")
    SVHN_test = f"{DRIVE_FOLDER}/SVHN_test.pkl"
    SVHN_train = f"{DRIVE_FOLDER}/SVHN_train.pkl"
    OUTPUT_DIR = DRIVE_FOLDER
elif os.path.exists("/kaggle/input"):
    print("Running on Kaggle")
    SVHN_test = "/kaggle/input/fii-atnn-2025-competition-2/SVHN_test.pkl"
    SVHN_train = "/kaggle/input/fii-atnn-2025-competition-2/SVHN_train.pkl"
    OUTPUT_DIR = "/kaggle/working"
else:
    print("Running locally")
    SVHN_test = "data/SVHN_test.pkl"
    SVHN_train = "data/SVHN_train.pkl"
    OUTPUT_DIR = "."

print(f"Training data: {SVHN_train}")
print(f"Test data: {SVHN_test}")
print(f"Output directory: {OUTPUT_DIR}")

class SVHN_Dataset(Dataset):
    def __init__(self, train: bool, transforms: v2.Transform, indices=None):
        path = SVHN_test
        if train:
            path = SVHN_train
        with open(path, "rb") as fd:
            all_data = pickle.load(fd)
        self.transforms = transforms

        if indices is not None:
            self.data = [all_data[i] for i in indices]
        else:
            self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        image, label = self.data[i]
        if self.transforms is None:
            return image, label
        return self.transforms(image), label


MEAN = [0.5070753693580627, 0.4865487813949585, 0.44091784954071045]
STD = [0.26733437180519104, 0.2564384937286377, 0.27615049481391907]

basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD, inplace=True)
])

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5), 
    v2.RandomCrop(32, padding=2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD, inplace=True)
])

num_classes = 100
cutmix_or_mixup = v2.RandomChoice([
    v2.CutMix(num_classes=num_classes, alpha=1.0),
    v2.MixUp(num_classes=num_classes, alpha=1.0),
])  

full_train_set = SVHN_Dataset(train=True, transforms=train_transforms)
val_set = SVHN_Dataset(train=True, transforms=basic_transforms)

train_size = int(0.9 * len(full_train_set))
indices = torch.randperm(len(full_train_set)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

print(f"Data split into: {train_size} train, {len(val_indices)} val")



def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    original_dtype = G.dtype
    X = G.float()
    
    norm = X.norm()
    if norm < eps:
        return torch.zeros_like(G).to(original_dtype)
    X = X / norm
    
    is_tall = G.size(0) > G.size(1)
    if is_tall:
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if is_tall:
        X = X.T
    
    return X.to(original_dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer - momentum with orthogonalized gradients for convolution weights.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g_update = g.add(buf, alpha=momentum)
                else:
                    g_update = buf

                # Orthogonalize for Conv layers only
                if len(g.shape) == 4 and p.data.shape[0] > 1:
                    g_2d = g_update.reshape(g_update.shape[0], -1)
                    g_ortho = zeropower_via_newtonschulz5(g_2d, steps=ns_steps)
                    g_update = g_ortho.view(g.shape)
                
                p.data.add_(g_update, alpha=-lr)

        return loss


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Classifier
            nn.Flatten(),
            nn.Linear(512, 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def run_experiment(name, optimizer_type, batch_size, lr, epochs, augmentation=True, tta=True, weight_decay=1.0, trans=train_transforms, label_smoothing=0.0, scheduler_type=None):
    print(f"\n{'='*60}")
    print(f"Version config: {name}")
    print(f"{'='*60}\n")

    start = time.time()

    train_trans = train_transforms if augmentation else basic_transforms
    train_set = SVHN_Dataset(train=True, transforms=trans, indices=train_indices)
    val_set = SVHN_Dataset(train=True, transforms=basic_transforms, indices=val_indices)
    test_set = SVHN_Dataset(train=False, transforms=basic_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=500, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=500, num_workers=4, pin_memory=True)

    model = VGG13().to(device)
    model = torch.jit.script(model)

    is_muon = optimizer_type == "muon_sgd"
    
    if optimizer_type == "muon_sgd":
        muon_params = []
        sgd_params = []
        
        for name_param, p in model.named_parameters():
            if len(p.shape) == 4 and 'weight' in name_param:
                muon_params.append(p)
            else:
                sgd_params.append(p)
        
        optimizer_muon = Muon(muon_params, lr=lr, momentum=0.95, nesterov=True)
        optimizer_sgd = optim.SGD(sgd_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        optimizer = {"muon": optimizer_muon, "sgd": optimizer_sgd}
        print(f"Using Muon for {len(muon_params)} conv params + SGD for {len(sgd_params)} other params")
    elif optimizer_type == "sgd":
        optimizer = {"main": optim.SGD(model.parameters(), lr=lr, fused=True)}
    elif optimizer_type == "sgd_momentum":
        optimizer = {"main": optim.SGD(model.parameters(), lr=lr, momentum=0.9)}
    elif optimizer_type == "sgd_momentum_nesterov":
        optimizer = {"main": optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)}
    elif optimizer_type == "adamw":
        optimizer = {"main": optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)}
    elif optimizer_type == "adam":
        optimizer = {"main": optim.Adam(model.parameters(), lr=lr)}
    elif optimizer_type == "rmsprop":
        optimizer = {"main": optim.RMSprop(model.parameters(), lr=lr)}

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scheduler = None
    if scheduler_type == "cosine_annealing":
        scheduler = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6) for opt in optimizer.values()]
    elif scheduler_type == "onecycle":
        scheduler = [optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 4, epochs=epochs, steps_per_epoch=len(train_loader)) for opt in optimizer.values()]
    elif scheduler_type == "cosine_annealing_restart":
        scheduler = []
        for opt in optimizer.values():
            warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=5)
            cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
            scheduler.append(optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[5]))
    elif scheduler_type == "plateau":
        main_opt = list(optimizer.values())[0]
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            main_opt,
            factor=0.5,
            patience=2,
            threshold=0.001,
            min_lr=1e-6,
        )

    def train():
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            original_targets = targets
            
            inputs, targets = cutmix_or_mixup(inputs, targets)
            
            if is_muon:
                # NO AMP for Muon
                for opt in optimizer.values():
                    opt.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                for opt in optimizer.values():
                    opt.step()
            else:
                for opt in optimizer.values():
                    opt.zero_grad()
                
                with torch.autocast(device.type, enabled=enable_half):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                for opt in optimizer.values():
                    scaler.step(opt)
                scaler.update()

            if scheduler is not None and scheduler_type == "onecycle":
                if isinstance(scheduler, list):
                    for sch in scheduler:
                        sch.step()
                else:
                    scheduler.step()

            predicted = outputs.argmax(1)
            total += original_targets.size(0)
            correct += predicted.eq(original_targets).sum().item()
            running_loss += loss.item() * inputs.size(0)

        if scheduler is not None and (scheduler_type == "cosine_annealing" or scheduler_type == "cosine_annealing_restart"):
            if isinstance(scheduler, list):
                for sch in scheduler:
                    sch.step()
            else:
                scheduler.step()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / total
        return accuracy, avg_loss

    @torch.inference_mode()
    def validate():
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with torch.autocast(device.type, enabled=enable_half and not is_muon):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item() * inputs.size(0)

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / total
        return accuracy, avg_loss

    @torch.inference_mode()
    def inference(use_tta=False):
        model.eval()
        labels = []

        tta_transforms = [basic_transforms]
        if use_tta:
            tta_transforms.append(v2.Compose([
                        v2.ToImage(),
                        v2.RandomHorizontalFlip(p=1.0),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(MEAN, STD, inplace=True)
                      
                    ]))
            tta_transforms.append(v2.Compose([
                    v2.ToImage(),
                    v2.CenterCrop(28),
                    v2.Resize(32),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(MEAN, STD, inplace=True)
                ]))
                    
        all_predictions = []
        for trans in tta_transforms:
            test_set_tta = SVHN_Dataset(train=False, transforms=trans)
            test_loader_tta = DataLoader(test_set_tta, batch_size=500, num_workers=4, pin_memory=True)

            batch_preds = []
            for inputs, _ in test_loader_tta:
                inputs = inputs.to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=enable_half and not is_muon):
                    outputs = model(inputs)
                batch_preds.append(outputs.cpu())

            all_predictions.append(torch.cat(batch_preds))
        avg_predictions = torch.stack(all_predictions).mean(dim=0)

        return avg_predictions.argmax(1).tolist()

    best_epoch = 0
    best_val_acc = 0.0

    print("\n" + "="*60)
    print("Training started")
    print("="*60)

    epochs_list = list(range(epochs))
    with tqdm(epochs_list) as tbar:
        for epoch in tbar:
            train_acc, train_loss = train()
            val_acc, val_loss = validate()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), f'{OUTPUT_DIR}/best_model_{name}.pth')

            if scheduler_type == "plateau" and scheduler is not None:
                scheduler.step(val_loss)

            current_lr = list(optimizer.values())[0].param_groups[0]['lr']
            tbar.set_description(f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Best: {best_val_acc:.2f}%, LR: {current_lr:.1e}")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{len(epochs_list)} | Train Acc: {train_acc:5.2f}% | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Loss Val: {val_loss:.4f} | Best: {best_val_acc:.2f}% at epoch {best_epoch+1} | LR: {current_lr:.1e}")

    print("="*60)
    print(f"Training done with the Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch+1}")
    print("="*60)

    end = time.time()

    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model_{name}.pth'))

    print("\n Predictions")
    data = {
        "ID": [],
        "target": []
    }

    for i, label in enumerate(inference(use_tta=tta)):
        data["ID"].append(i)
        data["target"].append(label)

    df = pd.DataFrame(data)
    df.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
    print(f"Submission saved to {OUTPUT_DIR}/submission.csv")
    print(f"Run duration: {(end - start) / 60:.2f} minutes")

    return {
        "name": name,
        "batch_size": batch_size,
        "run_duration": (end - start) / 60,
        "val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "optimizer": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        "augmentation": augmentation,
        "epochs": epochs,
        "tta": tta,
        "label_smoothing": label_smoothing,
        "scheduler_type": scheduler_type
    }


results = []



results.append(run_experiment(
    name="MUON_COSINE_FINAL",
    optimizer_type="muon_sgd",
    batch_size=256,
    lr=0.05,  
    epochs=150, 
    weight_decay=5e-4, 
    trans=train_transforms,
    label_smoothing=0.1,
    scheduler_type="cosine_annealing",  
    tta=True
))

df = pd.DataFrame(results)
print("\n" + "="*60)
print("="*60)
print(df.to_string(index=False))
df.to_csv(f'{OUTPUT_DIR}/experiment_results.csv', index=False)