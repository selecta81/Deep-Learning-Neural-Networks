import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from sklearn.metrics import *
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


def train_val(net, epoch, device, train_loader,valid_loader, writer, criterion, optimizer, 
             batch_size, classes, checkpoint_dir):
    
    train_loss = 0
    valid_loss = 0
    print(device)
    # training steps
    net.train()
    net.cuda()
    for batch_index, (data, target) in enumerate(train_loader):

        # moves tensors to GPU
        data, target =  data.cuda(), target.cuda()
       
        # clears gradients
        optimizer.zero_grad()
        # forward pass
        output = net(data)
        # loss in batch
        loss = criterion(output, target)
        # backward pass for loss gradient
        loss.backward()
        # update paremeters
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)

        if epoch == 0 and batch_index == 0:
            writer.add_graph(net, data.cuda())


    # save checkpoint with epoch index to checkpoint dir
    torch.save(net.state_dict(), os.path.join(checkpoint_dir,
                   "{}.pt".format(epoch)))
        
    # validation steps
    net.eval()
    for batch_index, (data, target) in enumerate(valid_loader):
        # moves tensors to GPU
        data, target = data.cuda(), target.cuda()

        # forward pass
        with torch.no_grad():          
            output = net(data)
        # loss in batch
        loss = criterion(output, target)
        # update validation loss
        valid_loss += loss.item()*data.size(0)
        
    # average loss calculations
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    
    # Display loss statistics
    print(f'Current Epoch: {epoch}\nTraining Loss: {round(train_loss, 6)}\nValidation Loss: {round(valid_loss, 6)}')
    writer.add_scalar("train/loss", train_loss, global_step = epoch)

    writer.add_scalar("val/loss", valid_loss, global_step = epoch)

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(valid_loader):    
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):               
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(10):      
        print('Val accuracy of %5s : %2d %%' % (
               classes[i], 100 * class_correct[i] / class_total[i]))
        writer.add_scalar("val/accuracy/{}".format(classes[i]), class_correct[i] / class_total[i],
                          epoch)
    
    writer.add_scalar("train/acc", round(100. * np.sum(class_correct) / np.sum(class_total), 2),
                     global_step = epoch)


def confusion_matrix_plot(net, device, valid_loader, writer, epoch, classes):
    
    net.eval()
    
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    
    confmat = confusion_matrix([i.item() for i in actuals], [i.item() for i in predictions])

    fig, ax = plot_confusion_matrix(conf_mat=confmat,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                class_names=classes)

    writer.add_figure("val/confusion_matrix",fig)

    return [i.item() for i in actuals], [i.item() for i in predictions]