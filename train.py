import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import functions
import utilities

import argparse
from utilities import data
from functions import build_classifier, train_model, accuracy_on_test, save_checkpoint

#adding arguments using the argparse module
parser = argparse.ArgumentParser(description='Neural network training.')

parser.add_argument('--data_directory', action='store', help='Path to training data'
parser.add_argument('--arch', action='store', required=False, dest = 'pretrained_model', default = 'vgg11',
                    help= 'Pretrained model to use, default: VGG-11.')
parser.add_argument('--save_dir', action = 'store', dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Location to save checkpoint.')
parser.add_argument('--dropout', action = 'store', dest='d_out', type=float, default = 0.5,
                    help = 'Dropout for model training, default: 0.5.')
parser.add_argument('--epochs', action='store', dest='n_epochs', type=int, default=3,
                    help = 'Number of epochs for training model, default: 3')
parser.add_argument('--learning_rate', action='store', dest=lr, type=float, default=0.001,
                    help = 'Learning rate while training model')
parser.add_argument('--print_every', action='store', dest='p_e', type=int, default=5,
                    help = 'Number of results to print')
parser.add_argument('--hidden', action='store', dest='hidden_layers', type=int, default=500,
                    help = 'Number of hidden layers, default: 500)

out_put = parser.parse_args()

data_dir = out_put.data_directory
save_dir = out_put.save_directory
dropout = out_put.d_out
epochs = out_put.n_epochs
learning_rate = out_put.lr
print_every = out_put.p_e
hidden = out_put.hidden_layers
trained_model = out_put.pretrained_model

#data processing
trainloader, testloader, validloader, train_set, test_set, valid_set = data(data_dir)

model = getattr(models,trained_model)(pretrained=True)

#build and attach new classifier
inputs = model.classifier[0].in_features
build_classifier(model, dropout, inputs, hidden)

#define criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
model.to(device)

#train model
model, optimizer = train_model(model,trainloader, validloader, criterion, optimizer, epochs, print_every)

#test model
accuracy_on_test(model,test_set, testloader, criterion)

#save model 
save_checkpoint(model, train_set, optimizer, save_dir, epochs)
