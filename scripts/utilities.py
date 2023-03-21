import os
import numpy as np
import json
import torch
import torchvision
import my_datasets
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_activations(activations_avg, ims = None, direction = None):
    if (ims is not None) and (direction is not None):
        activations = activations_avg[ims,:]
        if activations.ndim == 1:
            activations = activations[direction]
        else:
            activations = activations[:, direction]
    elif (direction is not None):
        activations = activations_avg[:,direction]
    elif (ims is not None): 
        activations = activations_avg[ims,:]
    return activations


def get_preds(paths, y, model, batch_start=0, batch_size=32):
    softmaxf = torch.nn.Softmax(dim=1)
    p1 = np.zeros((len(y),1000))
    while batch_start+batch_size < len(y)+batch_size: 
            # preprocessing the inputs 
            inputs = torch.stack([my_datasets.transform_normalize(my_datasets.transform(Image.open(paths[i]).convert("RGB"))) for i in range(batch_start, min(batch_start+batch_size, len(y)))])
            inputs = inputs.clone().detach().requires_grad_(True)
            batch_y=y[batch_start:min(batch_start+batch_size, len(y))]

            # transfering to GPU
            inputs=inputs.to(device)
            y1=model(inputs)

            p1[batch_start:min(batch_start+batch_size, len(y)),:]=softmaxf(y1).detach().cpu()
            batch_start+=batch_size
    return p1


def check_acc(preds, y):
    num_correct = 0
    for idx in range(len(y)):
        if np.argmax(preds[idx]) == y[idx]:
            num_correct+=1
    acc = num_correct / len(y)
    return acc