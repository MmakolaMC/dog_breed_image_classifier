import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

import argparse
from functions import load_checkpoint, predict 
from utilities import process_image

parser = argparse.ArgumentParser(description='Making predictions with model.')

parser.add_argument('--image_path', action='store', default = 'flowers/test/15/image_06369.jpg',
                    help='Path to image.')
parser.add_argument('--arch', action='store', dest = 'pretrained_model', default = 'vgg11',
                    help= 'Pretrained model to use, default: VGG-11.')
parser.add_argument('--save_dir', action = 'store', dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Location to save checkpoint.')
parser.add_argument('--cat_to_name', action='store', dest='ctn_dir', default = 'cat_to_name.json',
                    help='Path to image.')
parser.add_argument('--top_k', action='store', dest='topk', type=int, default = 5,
                    help='Top most likely classes a predicted flower can be, default: 5.')

out_put = parser.parse_args()

save_dir = out_put.save_directory
image = out_put.image_path
top_k = out_put.topk
cat_to_name = out_put.ctn_dir
trained_model = out_put.pretrained_model
with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f)
    
model = getattr(models,trained_model)(pretrained=True)

#load pretrained model
load_checkpoint(model, save_dir)

#image processing
process_image(image)

#predict classes and probabilities of classes
probs, classes = predict(model_loaded, image, topk)
print(probs)
print(classes)

Predicted_names = []
for i in classes:
    Predicted_names += [cat_to_name[i]]
    

print('Flower is most likely a {}, with a probability of {}'.format(Predicted_names[0], probs[0]))
