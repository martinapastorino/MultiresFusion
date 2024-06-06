# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 18:52:11 2022

@author: marti
"""

from IPython.display import clear_output
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn.init
from torch.autograd import Variable
from net.net import *
from utils.utils import *
from net.loss import *
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_dataset import *
from net.test_network import test
import cv2

 
def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch = 10):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss(weight=weights)
    iter_ = 0
    
    for e in tqdm(range(1, epochs + 1)):

        net.train()
        for batch_idx, (data_pan, data_hsi, target) in enumerate(train_loader):
            data_pan, data_hsi, target = Variable(data_pan.cuda()), Variable(data_hsi.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data_pan, data_hsi)[0]
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()  #loss.data[0]
            mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
            
            if iter_ % 100 == 0:
                clear_output()
                data = data_pan
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)))
                hsi = np.asarray(255 * np.transpose(data_hsi.data.cpu().numpy()[0],(1,2,0)))
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))
                plt.plot(mean_losses[:iter_]) and plt.show()
                fig = plt.figure(figsize=(10, 40))
                fig.add_subplot(141)
                plt.imshow(rgb)
                plt.title('RGB')
                fig.add_subplot(142)
                plt.imshow(cv2.resize(hsi[:,:,:3], dsize=(48, 48), interpolation=cv2.INTER_NEAREST))
                plt.title('HSI - 3 bands displayed')
                fig.add_subplot(143)
                plt.imshow(convert_to_color(gt))
                plt.title('Ground truth')
                fig.add_subplot(144)
                plt.title('Prediction')
                plt.imshow(convert_to_color(pred))
                plt.show()
            iter_ += 1
            
            del(data_pan, data_hsi, target, loss)

        if scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            torch.save(net.state_dict(), output_folder + 'test_epoch{}'.format(e))
    torch.save(net.state_dict(), output_folder + 'test_final')
    

def train_MS(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch = 10):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss(weight=weights)
    iter_ = 0
    
    for e in tqdm(range(1, epochs + 1)):

        net.train()

        for batch_idx, (data_pan, data_hsi, target) in enumerate(train_loader):
            data_pan, data_hsi, target = Variable(data_pan.cuda()), Variable(data_hsi.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data_pan, data_hsi)[0]
            loss_C = CrossEntropy2d(output, target, weight=weights)
            out = torch.clamp(output[:, 0:11], 1e-10, 1.0)   #num classes
            loss_L = levelsetLoss()(out, data_pan)
            loss_A = gradientLoss2d()(out) * 0.001
            loss_LS = (loss_L + loss_A) * lambda_A

            loss = loss_C + loss_LS

            loss.backward()
            optimizer.step()


            losses[iter_] = loss.item()  #loss.data[0]
            mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
            
            if iter_ % 100 == 0:
                clear_output()
                data = data_pan
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0],(1,2,0)))
                hsi = np.asarray(255 * np.transpose(data_hsi.data.cpu().numpy()[0],(1,2,0)))
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))
                plt.plot(mean_losses[:iter_]) and plt.show()
                fig = plt.figure(figsize=(10, 40))
                fig.add_subplot(141)
                plt.imshow(rgb)
                plt.title('RGB')
                fig.add_subplot(142)
                plt.imshow(cv2.resize(hsi[:,:,:3], dsize=(48, 48), interpolation=cv2.INTER_NEAREST))
                plt.title('HSI - 3 bands displayed')
                fig.add_subplot(143)
                plt.imshow(convert_to_color(gt))
                plt.title('Ground truth')
                fig.add_subplot(144)
                plt.title('Prediction')
                plt.imshow(convert_to_color(pred))
                plt.show()
            iter_ += 1
            
            
            del(data_pan, data_hsi, target, loss)

        if scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            torch.save(net.state_dict(), output_folder + 'test_epoch{}'.format(e))
    torch.save(net.state_dict(), output_folder + 'test_final')