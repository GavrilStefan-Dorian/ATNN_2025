import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# !nvidia-smi # test we are on GPU

# %pip uninstall -y torch torchaudio torchvision
# %pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/test/cu126 -q # 2.9.1 has Muon
# %pip install torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/test/cu126 -q 

# %pip install torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126 -q
# #
# %pip install timm wandb==0.22.0 torchmetrics numpy tensorboard matplotlib -q 

# %pip install argparse yaml -q
import os

import argparse
import yaml

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

from torchvision.datasets import CIFAR10, CIFAR100, MNIST, OxfordIIITPet
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import timm


# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("wandb_key")
# import wandb
# wandb.login(key=secret_value_0)
import wandb
wandb.login(key=os.getenv('WANDB_API_KEY')) # kaggle uses kaggle_secrets, here use .env

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


print("Muon" in dir(torch.optim))
print(torch.__version__)
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
enable_half = device.type != "cpu"
# scaler = torch.cuda.amp.GradScaler(device, enabled=enable_half)

print("Grad scaler is enabled:", enable_half)
print("Device:", device)

batch_size_map = {
    "resnet18": 128,
    "resnet50": 64,
    "resnest14d": 32,
    "resnest26d": 16,
    "mlp": 256
}   
class BatchSizeScheduler:
    def __init__(self, initial_batch_size, max_batch_size, step_size=30):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.step_size = step_size
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0 and self.current_batch_size < self.max_batch_size:
            old_bs = self.current_batch_size
            self.current_batch_size = min(int(self.current_batch_size * 1.5), self.max_batch_size)
            if self.current_batch_size != old_bs:
                print(f"Batch size increased: {old_bs} → {self.current_batch_size}")
                return True
        return False
    
    def get_batch_size(self):
        return self.current_batch_size
def get_transforms(dataset, image_size, is_train=True, use_heavy_aug=False):
    if dataset == "MNIST":
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.ToDtype(torch.float32, scale=True),
        ])
    
    if dataset == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == "CIFAR100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2009, 0.1984, 0.2023)
    else:  # OxfordIIITPet
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    if is_train:
        if image_size == 32:
            transforms_list = [
                v2.ToImage(),
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
            ]
            if use_heavy_aug:
                transforms_list.append(v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
            transforms_list.extend([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std)
            ])
            return v2.Compose(transforms_list)
        else:  # For pretrained models (224x224)
            transforms_list = [
                v2.ToImage(),
                v2.Resize(256),
                v2.RandomCrop(image_size),
                v2.RandomHorizontalFlip(),
            ]
            transforms_list.extend([
                v2.ColorJitter(0.4, 0.4, 0.4, 0.1),  
                v2.RandomErasing(p=0.25),            
            ])
            transforms_list.extend([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std)
            ])
            return v2.Compose(transforms_list)
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.CenterCrop(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std)
        ])

# DATA_ROOT = "kaggle/working/data"
# os.makedirs('./data', exist_ok=True)

# in kaggle use DATA_ROOT and replace root='./data' 
# also replace download with =not os.path.exists(f"{DATA_ROOT}/....")

DATA_ROOT = os.path.join(os.getcwd(), 'data')
os.makedirs(DATA_ROOT, exist_ok=True)

def get_data_loaders(dataset_name, image_size, batch_size, num_workers=2, pin_memory=True, use_heavy_aug=False):
    train_transforms = get_transforms(dataset_name, image_size, is_train=True, use_heavy_aug=use_heavy_aug)
    test_transform = get_transforms(dataset_name, image_size, is_train=False)

    if dataset_name == "CIFAR100":
        train_dataset= CIFAR100(root=DATA_ROOT, train=True, download=True, transform=train_transforms)
        test_dataset = CIFAR100(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    elif dataset_name == "CIFAR10":
        train_dataset= CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transforms)
        test_dataset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    elif dataset_name == "MNIST":
        train_dataset= MNIST(root=DATA_ROOT, train=True, download=True, transform=train_transforms)
        test_dataset = MNIST(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    elif dataset_name == "OxfordIIITPet":
        train_dataset= OxfordIIITPet(root=DATA_ROOT, download=True, transform=train_transforms)
        test_dataset = OxfordIIITPet(root=DATA_ROOT, split='test', download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
def get_num_classes(dataset_name):
    if dataset_name in ["CIFAR100"]:
        return 100
    elif dataset_name in ["CIFAR10"]:
        return 10
    elif dataset_name in ["MNIST"]:
        return 10
    elif dataset_name in ["OxfordIIITPet"]:
        return 37
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
ALLOWED_MODELS = ["resnet18", "resnet50", "resnest14d", "resnest26d", "mlp"]

def create_model(model_name, dataset_name, pretrained=False, image_size=32):
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    
    num_classes=get_num_classes(dataset_name)

    if model_name == "mlp":
        in_channels = 1 if dataset_name == "MNIST" else 3
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * image_size * image_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    else:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if not pretrained and image_size == 32:
            if model_name.startswith("resnet") or model_name.startswith("resnest"):
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
    return model
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho
        self.adaptive = adaptive

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Base optimizer instance
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e)  # ascent step
                self.state[p]["e"] = e

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e"])  # restore weights

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self):
        # Not used — SAM requires first_step + second_step manually
        raise RuntimeError("Use first_step() and second_step() with SAM.")
    
    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                ((torch.pow(p, 2) if self.adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def create_optimizer(optimizer_name, model, lr, weight_decay, pretrained):
    if optimizer_name == "SGD" and pretrained:
        param_groups = [
            {"params": [], "lr": lr * 0.25},
            {"params": [], "lr": lr * 0.5},
            {"params": [], "lr": lr * 1.0},
        ]

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "layer4" in name or "fc" in name:
                param_groups[2]["params"].append(p)
            elif "layer3" in name:
                param_groups[1]["params"].append(p)
            else:
                param_groups[0]["params"].append(p)

        return optim.SGD(param_groups, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    
    elif optimizer_name == "SGD" and not pretrained:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SAM":
        return SAM(model.parameters(), base_optimizer=optim.SGD, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "Muon":
        return torch.optim.Muon(model.parameters(), lr=lr, weight_decay=weight_decay)        
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
def create_scheduler(scheduler_name, optimizer, step_size=30):
    base_optimizer = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer

    if scheduler_name == "StepLR":
        return StepLR(base_optimizer, step_size=step_size, gamma=0.1)
    elif scheduler_name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(base_optimizer, patience=5)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, 
                mixup_cutmix_transform=None):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if mixup_cutmix_transform is not None:
            inputs, targets = mixup_cutmix_transform(inputs, targets)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if isinstance(optimizer, SAM):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer.base_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.first_step(zero_grad=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer.base_optimizer)
            scaler.step(optimizer.base_optimizer)
            scaler.update()
            optimizer.zero_grad()
        else: 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if targets.dim() > 1:  # Mixed labels (one-hot/soft labels)
            correct += predicted.eq(targets.argmax(1)).sum().item()
        else:  # Regular labels
            correct += predicted.eq(targets).sum().item()

    return train_loss / len(train_loader), 100.0 * correct/total

def test(model, test_loader, criterion, device, use_amp):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / len(test_loader), 100.0 * correct/total

def train_model(config):
    label_smoothing = config.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=config["use_amp"])
    writer = SummaryWriter(log_dir=f'./logs/{config["dataset"]}/{config["optimizer"]}_{config["model"]}_bs{batch_size_map[config["model"]]}_lr{config["lr"]}_sched{config["scheduler"]}')

    image_size = 224 if config["pretrained"] else 32
    initial_batch_size = batch_size_map[config["model"]]
    
    use_bs_scheduler = config.get("use_batch_size_scheduler", False)
    if use_bs_scheduler:
        bs_scheduler = BatchSizeScheduler(
            initial_batch_size=initial_batch_size // 2, 
            max_batch_size=initial_batch_size,
            step_size=30
        )
        batch_size = bs_scheduler.get_batch_size()
        print(f"Batch size scheduler enabled: {batch_size} → {initial_batch_size}")
    else:
        batch_size = initial_batch_size

    use_heavy_aug = config.get("use_heavy_aug", False)
    train_loader, test_loader = get_data_loaders(config["dataset"], image_size, batch_size, 
                                                 config["num_workers"], config["pin_memory"], 
                                                 use_heavy_aug=use_heavy_aug) 
    model = create_model(config["model"], config["dataset"], config["pretrained"], image_size)
    model.to(device)

    if config.get("use_compile", False) and hasattr(torch, "compile"):
        print("Compiliing is enabled")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Compiling failed: {e}.")

    # Note: now passes model instead of model.parameters() for layer-wise LR
    optimizer = create_optimizer(config["optimizer"], model, config["lr"], config["weight_decay"], config["pretrained"])
    
    step_size = config.get("step_size", 30)
    scheduler = create_scheduler(config["scheduler"], optimizer, step_size=step_size)

    epochs = config["epochs"]
    
    best_acc = 0.0
    best_acc_epoch = None
    counter = 0
    patience = config.get("patience", 10)

    mixup_cutmix_transform = None
    use_mixup = config.get("use_mixup", False)
    use_cutmix = config.get("use_cutmix", False)
    
    if use_mixup or use_cutmix:
        num_classes = get_num_classes(config["dataset"])
        transforms_list = []
        
        if use_mixup:
            mixup_alpha = config.get("mixup_alpha", 0.2)
            transforms_list.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
        
        if use_cutmix:
            cutmix_alpha = config.get("cutmix_alpha", 0.2)
            transforms_list.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
        
        if len(transforms_list) > 1:
            mixup_cutmix_transform = v2.RandomChoice(transforms_list)
        else:
            mixup_cutmix_transform = transforms_list[0]

    start_time = time.time()
    print("Start training")
    
    for epoch in range(epochs):
        epoch_start = time.time()

        if use_bs_scheduler and bs_scheduler.step():
            batch_size = bs_scheduler.get_batch_size()
            train_loader, test_loader = get_data_loaders(
                config["dataset"], 
                image_size, 
                batch_size,
                config.get("num_workers", 2),
                config.get("pin_memory", True),
                use_heavy_aug=use_heavy_aug
            )

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, 
                                           device, config["use_amp"],
                                           mixup_cutmix_transform=mixup_cutmix_transform)
        test_loss, test_acc = test(model, test_loader, criterion, device, config["use_amp"])

        if isinstance(scheduler, StepLR):
            scheduler.step()
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(test_loss)
        
        if isinstance(optimizer, SAM):
            current_lr = optimizer.base_optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        if config["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": test_loss,
                "val_acc": test_acc,
                "lr": current_lr,
                "epoch_time": epoch_time,
                "batch_size": batch_size if use_bs_scheduler else initial_batch_size
            })

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        if use_bs_scheduler:
            writer.add_scalar('Batch_Size', batch_size, epoch)
        
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs('./checkpoints', exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,    
                'config': config    
            }, f'./checkpoints/{config["optimizer"]}_{config["model"]}_bs{batch_size_map[config["model"]]}_lr{config["lr"]}_sched{config["scheduler"]}.pth')
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

        if best_acc_epoch is None:
            best_acc_epoch = test_acc
        elif test_acc <= best_acc_epoch:  
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            best_acc_epoch = test_acc
            counter = 0

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    writer.close()

    if config["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "train_acc": train_acc,
                "lr": current_lr,
                "epoch_time": epoch_time,
                "batch_size": batch_size if use_bs_scheduler else initial_batch_size
            })

    return {
        'best_acc': best_acc,
        'total_time': total_time,
        'final_train_acc': train_acc,
        'final_test_acc': test_acc
    }

def create_sweep_config(pretrained=False):
    if pretrained:
        return {
            "method": "grid",
            "metric": {"name": "val_acc", "goal": "maximize"},
            "parameters": {
                "dataset": {"value": "CIFAR100"},
                "model": {"values": ["resnet18", "resnet50"]},
                "pretrained": {"value": True}, 
                "lr": {"values": [3e-4, 1e-4]},
                "optimizer": {"value": "AdamW"},  
                "scheduler": {"value": "StepLR"},  
                "epochs": {"value": 100},
                "weight_decay": {"value": 1e-2},
                "num_workers": {"value": 2},  
                "pin_memory": {"value": True},
                "use_amp": {"value": True},  
                "use_wandb": {"value": True},
                "use_batch_size_scheduler": {"value": False}
            }
        }
    else:
        return {
            "method": "grid",
            "metric": {"name": "val_acc", "goal": "maximize"},
            "parameters": {
                "dataset": {"value": "CIFAR100"},
                "model": {"values": ["resnet18", "resnet50"]},
                "pretrained": {"value": False}, 
                "lr": {"values": [0.1, 0.05]},  
                "optimizer": {"value": "SGD"}, 
                "scheduler": {"values": ["StepLR", "ReduceLROnPlateau"]},
                "epochs": {"value": 200},
                "weight_decay": {"value": 5e-4},
                "num_workers": {"value": 2}, 
                "pin_memory": {"value": True},
                "use_amp": {"value": True}, 
                "use_wandb": {"value": True},
                "use_batch_size_scheduler": {"value": False}
            }
        }
def sweep_train():
    run = wandb.init(project="training_pipeline")

    cfg = dict(wandb.config)
    results = train_model(cfg)
    run.finish()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML sweep configuration file.")
    parser.add_argument("--override", nargs="*", default=[], help="Override configuration parameters in the format key=value.")
    return parser.parse_args()

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_override_list(override_list):
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}")
        key, val = item.split("=", 1)

        # try to auto-convert to float/int/bool/list
        if val.lower() in ["true", "false"]:
            val = val.lower() == "true"
        else:
            try:
                val = int(val)
            except:
                try:
                    val = float(val)
                except:
                    pass  # keep string

        overrides[key] = val
    return overrides

def parse_value(x):
    x = x.strip()
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            if x.lower() in ["true", "false"]:
                return x.lower() == "true"
            return x
        
def apply_cli_overrides(sweep_cfg, overrides):
    params = sweep_cfg.get("parameters", {})

    for key, new_val in overrides.items():
        if key not in params:
            print(f"Override ignored! (not in sweep parameters): {key}")
            continue

        #"value": X
        if "value" in params[key]:
            params[key]["value"] = new_val

        # "values": [list]
        elif "values" in params[key]:

            if isinstance(new_val, str) and "," in new_val:
                new_val = [parse_value(x) for x in new_val.split(",")]
            else:
                new_val = [new_val]  
            params[key]["values"] = new_val

    return sweep_cfg

if __name__ == "__main__":
    # sweep_config = create_sweep_config(pretrained=False) # for hardcoded runs

    args = parse_args()
    config_path = args.config    

    overrides = parse_override_list(args.override) # get the overrides from CLI             
    sweep_config = load_yaml_config(config_path)
    if overrides:
        sweep_config = apply_cli_overrides(sweep_config, overrides)

    sweep_id = wandb.sweep(sweep_config, project="training_pipeline")
    wandb.agent(sweep_id, function=sweep_train)
