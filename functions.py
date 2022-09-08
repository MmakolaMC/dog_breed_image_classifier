import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from time import time, sleep, localtime, strftime
from collections import OrderedDict

import utilities
from utilities import data, process_image

#function to build a new classifier
def build_classifier(model,inputs, hidden, dropout):
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                               ('fc1', nn.Linear(inputs)),
                               ('relu', nn.ReLU()),
                               ('dropout1', nn.Dropout(dropout)),
                               ('fc2', nn.Linear(hidden)),
                               ('output', nn.LogSoftmax(dim=1))
                                ]))
    model.classifier = classifier
    return model

#cuda modes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

#function to train model
def train_model(model,trainloader, validloader, criterion, optimizer, epochs, print_every):
    with active_session():
        
        steps = 0
        running_loss = 0
    
        start_time = time()
        for epoch in range(epochs):
            for images, labels in trainloader:
                steps +=1
        
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()#zero gradients to save computation time
        
                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
        
                running_loss = loss.item()
            
                if steps % print_every ==0:
                    model.eval()
                    validation_loss = 0
                    accuracy = 0
                    with torch.no_grad():    
                        for images, labels in validloader:
            
                            images, labels = images.to(device), labels.to(device)
        
                            logps = model(images)
                            loss = criterion(logps, labels)
                            validation_loss +=loss.item()
        
                            ps = torch.exp(logps)
                            top_ps, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                        
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")    
                    running_loss = 0
                    model.train()
                    
        tot_time = time() - start_time
        tot_time = strftime('%H:%M:%S', localtime(tot_time))
        print("\n** Total Elapsed Training Runtime: ", tot_time)
        
        return model, optimizer

#function to test model accuracy
def accuracy_on_test(model,test_set, testloader,criterion):  
    val_loss = 0
    accuracy = 0
    #test model with cuda
    model.to(device)
    
    #turn off drop-out
    model.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            loss = criterion(logps, labels)
            val_loss +=loss.item()
        
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    print('Accuracy of the network on {} test images: {:.3f}%'.format(len(test_set), accuracy/len(testloader) * 100))

#function to save checkpoint/model
def save_checkpoint(model, train_set, optimizer, save_dir, epochs):
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier':model.classifier,
                  'epochs': epochs,
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx':train_set.class_to_idx}

    return torch.save(checkpoint, save_dir)

#function to load the saved checkpoint/model
def load_checkpoint(model, save_dir):
    if device:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_loaction=lambda storage, loc:storage)
    
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

#function to predict classes using saved model
def predict( model_loaded, image, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if device:
        model_loaded.to(device)
    else:
        model_loaded.cpu()
    
    image = process_image(image)
    
    #convert image to tensor
    image =torch.from_numpy(image).type(torch.cuda.FloatTensor)
    
    image = image.unsqueeze(0)
    
    out_p = model(image)
    
    ps = torch.exp(out_p)
    
    #probabilities and indices corresponding to classes
    top_ps, top_idcs = ps.topk(topk)
    
    #convert to lists
    top_ps = top_ps.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_idcs = top_idcs.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    idx_to_class = {value: key for key, value in model_loaded.class_to_idx.items()}
    
    top_classes = [idx_to_class[index] for index in top_idcs]
                   
    
    return top_ps, top_classes
