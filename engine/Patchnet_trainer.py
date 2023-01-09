import os
from random import randint
import torch
import torchvision
import torch.nn.functional as F
from engine.base_trainer import BaseTrainer
from metrics.meter import AvgMeter
from tqdm import tqdm
import time
from utils.utils import calc_acc

class Trainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, loss, lr_scheduler, device, trainloader, valloader, writer):
        super(Trainer, self).__init__(cfg, network, optimizer, loss, lr_scheduler, device, trainloader, valloader, writer)
        self.network = self.network.to(device)
        
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))

    def load_model(self):
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.\
            format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        state = torch.load(saved_name)

        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])
        self.loss.load_state_dict(state['loss'])
        
    def save_model(self, epoch, val_loss):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}_{}.pth'.\
            format(self.cfg['model']['base'], epoch, val_loss))

        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss.state_dict()
        }
        
        torch.save(state, saved_name)
        
    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        for i, (img1, img2, label) in enumerate(self.trainloader):
            img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
            feature1 = self.network(img1)
            feature2 = self.network(img2)
            self.optimizer.zero_grad()
            loss = self.loss(feature1, feature2, label)
            loss.backward()
            self.optimizer.step()

            score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature1.squeeze()), dim=1)
            score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature2.squeeze()), dim=1)

            acc1 = calc_acc(score1, label.squeeze().type(torch.int32))
            acc2 = calc_acc(score2, label.squeeze().type(torch.int32))
            accuracy = (acc1 + acc2) / 2
            
            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)

            print('Epoch: {:3}, iter: {:5}, loss: {:.5}, acc: {:.5}'.\
                format(epoch, epoch * len(self.trainloader) + i, \
                self.train_loss_metric.avg, self.train_acc_metric.avg))
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
    def train(self):
        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            epoch_loss = self.validate(epoch)
            # if epoch_acc > self.best_val_acc:
            #     self.best_val_acc = epoch_acc
            self.save_model(epoch, epoch_loss)
            
    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        seed = randint(0, len(self.valloader)-1)
        
        with torch.no_grad():
            for i, (img1, img2, label) in enumerate(self.valloader):
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
                feature1 = self.network(img1)
                feature2 = self.network(img2)
                loss = self.loss(feature1, feature2, label)

                score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature1.squeeze()), dim=1)
                score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature2.squeeze()), dim=1)

                acc1 = calc_acc(score1, label.squeeze().type(torch.int32))
                acc2 = calc_acc(score2, label.squeeze().type(torch.int32))
                accuracy = (acc1 + acc2) / 2

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)
        
        print("Validation epoch {} =============================".format(epoch))
        print("Epoch: {:3}, loss: {:.5}, acc: {:.5}".format(epoch, self.val_loss_metric.avg, self.val_acc_metric.avg))
        print("=================================================")

        return self.val_loss_metric.avg
                