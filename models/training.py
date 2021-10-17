from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
from time import time


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer(object):
    def __init__(self, encoder, decoder, cfg, device, train_dataset, val_dataset, exp_name, optimizer='Adam'):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        print('Number of Parameters in encoder: {}'.format(count_parameters(self.encoder)))
        print('Number of Parameters in decoder: {}'.format(count_parameters(self.decoder)))
        self.cfg = cfg
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(params=list(encoder.parameters())+list(decoder.parameters()),
                                        lr=self.cfg['lr'])
        self.lr = self.cfg['lr']

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None

        self.criterion = torch.nn.BCEWithLogitsLoss()


    def reduce_lr(self, epoch):
        if epoch > 0 and self.cfg['lr_decay_interval'] is not None and epoch % self.cfg['lr_decay_interval'] == 0:
            self.lr = self.lr * self.cfg['lr_decay_factor']
            print("Reducing LR to {}".format(self.lr))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

    def train_step(self, batch):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        if self.cfg['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=self.cfg['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.cfg['grad_clip'])
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, batch):
        device = self.device

        points = batch.get('points').to(device)
        occ = batch.get('occupancies').to(device).squeeze()
        inputs = batch.get('inputs').to(device)

        encoding = self.encoder(inputs)
        logits = self.decoder(points, encoding).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, occ.squeeze(), reduction='mean')

        return loss

    def train_model(self, epochs, ckp_interval=1):
        loss = 0
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            self.reduce_lr(epoch)
            sum_loss = 0
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            for batch in train_data_loader:
                loss = self.train_step(batch)
                sum_loss += loss


            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)

            if epoch % ckp_interval == 0:
                self.save_checkpoint(epoch)
                val_loss = self.compute_val_loss()

                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch),[epoch,val_loss])

                self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                print("Epoch {:5d}: Train Loss: {:06.4f}, Val Loss: {:06.4f}".format(epoch, sum_loss/len(train_data_loader), val_loss))

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):

            torch.save({'epoch': epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'decoder_state_dict': self.decoder.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        if self.cfg['ckpt'] is not None:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(self.cfg['ckpt'])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        epoch += 1
        for param_group in self.optimizer.param_groups:
            print('Setting LR to {}'.format(self.cfg['lr']))
            param_group['lr'] = self.cfg['lr']
        if self.cfg['lr_decay_interval'] is not None:
            decay_steps = int(epoch/self.cfg['lr_decay_interval'])
            lr = self.cfg['lr'] * self.cfg['lr_decay_factor']**decay_steps
            print('Reducting LR to {}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr * self.cfg['lr_decay_factor']**decay_steps
        return epoch

    def compute_val_loss(self):
        self.encoder.eval()
        self.decoder.eval()

        val_data_loader = self.val_dataset.get_loader()
        sum_val_loss = 0
        for val_batch in val_data_loader:
            sum_val_loss += self.compute_loss(val_batch).item()


        return sum_val_loss / len(val_data_loader)
