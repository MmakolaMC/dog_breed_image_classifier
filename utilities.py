import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image

from torchvision import datasets, transforms, models

#data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#function to load and process flowers data
def data():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_set = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=32)
    
    return trainloader, testloader, validloader, train_set, test_set, valid_set

#funtion to process test image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Done: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    imageprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    image_tensor = imageprocess(image)
    
    return image_tensor.numpy()