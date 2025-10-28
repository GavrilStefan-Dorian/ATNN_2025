import os
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import v2
from torch import GradScaler, optim
from tqdm import tqdm
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


class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


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
    v2.RandomCrop(32, padding=4),
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


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512, 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def run_experiment(name, batch_size, lr, epochs, weight_decay, use_ema, ema_decay, pct_start):
    start = time.time()

    train_set = SVHN_Dataset(train=True, transforms=train_transforms, indices=train_indices)
    val_set = SVHN_Dataset(train=True, transforms=basic_transforms, indices=val_indices)
    test_set = SVHN_Dataset(train=False, transforms=basic_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=500, num_workers=4, pin_memory=True)

    model = VGG13().to(device)
    model = torch.jit.script(model)

    ema = EMA(model, decay=ema_decay) if use_ema else None

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
        anneal_strategy='cos',
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
            
            with torch.autocast(device.type, enabled=enable_half):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()
            
            if ema:
                ema.update()

            predicted = outputs.argmax(1)
            total += original_targets.size(0)
            correct += predicted.eq(original_targets).sum().item()
            running_loss += loss.item() * inputs.size(0)

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

            with torch.autocast(device.type, enabled=enable_half):
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
    def inference():
        model.eval()
        
        tta_transforms = [
            basic_transforms,
            v2.Compose([
                v2.ToImage(),
                v2.RandomHorizontalFlip(p=1.0),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(MEAN, STD, inplace=True)
            ]),
            v2.Compose([
                v2.ToImage(),
                v2.CenterCrop(28),
                v2.Resize(32),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(MEAN, STD, inplace=True)
            ])
        ]
                    
        all_predictions = []
        for trans in tta_transforms:
            test_set_tta = SVHN_Dataset(train=False, transforms=trans)
            test_loader_tta = DataLoader(test_set_tta, batch_size=500, num_workers=4, pin_memory=True)

            batch_preds = []
            for inputs, _ in test_loader_tta:
                inputs = inputs.to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=enable_half):
                    outputs = model(inputs)
                batch_preds.append(outputs.cpu())

            all_predictions.append(torch.cat(batch_preds))
        
        avg_predictions = torch.stack(all_predictions).mean(dim=0)
        return avg_predictions.argmax(1).tolist()

    best_epoch = 0
    best_val_acc = 0.0

    epochs_list = list(range(epochs))
    with tqdm(epochs_list) as tbar:
        for epoch in tbar:
            train_acc, train_loss = train()
            val_acc, val_loss = validate()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
                if ema:
                    ema.apply_shadow()
                if ema:
                    ema.restore()

            current_lr = optimizer.param_groups[0]['lr']
            tbar.set_description(f"Val: {val_acc:.2f}%, Best: {best_val_acc:.2f}% @ {best_epoch+1}, LR: {current_lr:.2e}")

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")


    end = time.time()


    data = {"ID": [], "target": []}
    for i, label in enumerate(inference()):
        data["ID"].append(i)
        data["target"].append(label)

    df = pd.DataFrame(data)
    df.to_csv(f'{OUTPUT_DIR}/submission_{name}.csv', index=False)

    return {
        "name": name,
        "use_ema": use_ema,
        "ema_decay": ema_decay if use_ema else None,
        "pct_start": pct_start,
        "weight_decay": weight_decay,
        "val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "run_duration": (end - start) / 60,
    }



if __name__ == "__main__":
    results = []
    
    base_config = {
        "batch_size": 128,
        "lr": 0.001,
        "epochs": 90,
    }
    
    results.append(run_experiment(
        name="no_ema",
        use_ema=False,
        ema_decay=None,
        pct_start=0.25,
        weight_decay=0.01,
        **base_config
    ))
    
    results.append(run_experiment(
        name="ema_0995",
        use_ema=True,
        ema_decay=0.995,
        pct_start=0.15,
        weight_decay=0.005,
        **base_config
    ))
    
    results.append(run_experiment(
        name="ema_099",
        use_ema=True,
        ema_decay=0.99,
        pct_start=0.15,
        weight_decay=0.005,
        **base_config
    ))
    
    # 1clr was 'improved'
    results.append(run_experiment(
        name="no_ema_improved",
        use_ema=False,
        ema_decay=None,
        pct_start=0.15,
        weight_decay=0.005,
        **base_config
    ))
    
    df = pd.DataFrame(results)
    df = df.sort_values('val_acc', ascending=False)
    
    
    best = df.iloc[0]
   
    df.to_csv(f'{OUTPUT_DIR}/ema_comparison.csv', index=False)