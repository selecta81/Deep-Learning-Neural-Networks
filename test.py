import torch
from sklearn.metrics import *
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def test(net, criterion, test_loader,device, writer, classes):
    
    # tracking test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    net.eval()

    for batch_idx, (data, target) in enumerate(test_loader):
        # move tensors to GPU
        data, target = data.cuda(), target.cuda()

        with torch.no_grad():            
            # forward pass
            output = net(data)
        # batch loss
        loss = criterion(output, target)
        # test loss update
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(4):
           label = target.data[i]
           class_correct[label] += correct[i].item()
           class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print(f'Test Loss: {round(test_loss, 6)}')

    for i in range(10):
        if class_total[i] > 0:
           print(f'Test Accuracy of {classes[i]}: {round(100*class_correct[i]/class_total[i], 2)}%')
        else:
           print(f'Test Accuracy of {classes[i]}s: N/A (no training examples)')
        
        
    print(f'Full Test Accuracy: {round(100. * np.sum(class_correct) / np.sum(class_total), 2)}% {np.sum(class_correct)} out of {np.sum(class_total)}')
