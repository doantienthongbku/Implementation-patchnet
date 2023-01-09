import os
import sys
import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from engine.Patchnet_trainer import Trainer
from metrics.losses import PatchLoss
from dataset.FAS_dataset import FASDataset
from utils.utils import read_cfg, get_optimizer, build_network, \
    get_device, get_rank

cfg = read_cfg(cfg_file='config/config.yaml')

# fix the seed for reproducibility
seed = cfg['seed'] + get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

# build model and engine
device = get_device(cfg)
model = build_network(cfg, device)
model.to(device)
optimizer = get_optimizer(cfg, model)
lr_scheduler = StepLR(optimizer=optimizer, step_size=90, gamma=0.5)
criterion = PatchLoss().to(device=device)
writer = SummaryWriter(cfg['log_dir'])

# dump_input = torch.randn((1, 3, cfg['dataset']['augmentation']['rand_crop_size'], cfg['dataset']['augmentation']['rand_crop_size']))
# writer.add_graph(model, dump_input)

# Without Resize transform, images are of different sizes and it causes an error
train_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.RandomCrop(cfg['dataset']['augmentation']['rand_crop_size']),
    transforms.RandomHorizontalFlip(cfg['dataset']['augmentation']['rand_hori_flip']),
    transforms.RandomRotation(cfg['dataset']['augmentation']['rand_rotation']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size']),
    transforms.RandomCrop(cfg['dataset']['augmentation']['rand_crop_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

trainset = FASDataset(
    root_dir=cfg['dataset']['root'],
    transform=train_transform,
    csv_file=cfg['dataset']['train_set'],
    is_train=True
)

valset = FASDataset(
    root_dir=cfg['dataset']['root'],
    transform=val_transform,
    csv_file=cfg['dataset']['val_set'],
    is_train=False
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=4
)

valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=cfg['val']['batch_size'],
    shuffle=True,
    num_workers=4
)

trainer = Trainer(
    cfg=cfg,
    network=model,
    optimizer=optimizer,
    loss=criterion,
    lr_scheduler=lr_scheduler,
    device=device,
    trainloader=trainloader,
    valloader=valloader,
    writer=writer
)

print("Start training...")
trainer.train()

writer.close()
