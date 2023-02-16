# some functions from discovery/scripts/cdisco/cdisco.py

import numpy as np
import torch
import torchvision 
import PIL.Image as Image
from datasets import transform
from datasets import transform_normalize

def get_model_state(model, paths, y, dim_c, dim_w, dim_h, SAVEFOLD=''):
    batch_size = 32
    tot_acc = 0
    i=0
    batch_start=0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    embeddings = np.zeros((len(y),2048))
    gradients = np.zeros((len(y), 2048))
    predictions = np.zeros((len(y), 1000))
    conv_embeddings=np.zeros((len(y)
                              , dim_c))
    gradients_wrt_conv_layer = np.zeros((len(y), dim_c, dim_w, dim_h), dtype=np.float32)
    conv_maps = np.zeros((len(y),dim_c,dim_w,dim_h))

    print(f"embeddings shape: {embeddings.shape}")
    print(f"gradients shape: {gradients.shape}")
    print(f"predictions shape: {predictions.shape}")

    while batch_start+batch_size < len(y)+batch_size: 
        # preprocessing the inputs 
        print(batch_start)
        inputs = torch.stack([transform_normalize(transform(Image.open(paths[i]).convert("RGB"))) for i in range(batch_start, min(batch_start+batch_size, len(y)))])
        inputs = inputs.clone().detach().requires_grad_(True)
        batch_y=y[batch_start:min(batch_start+batch_size, len(y))]

        # transfering to GPU
        inputs=inputs.to(device)
        model=model.to(device)

        # inference pass
        outs = model(inputs)

        # extracting embeddings
        # note: convolutional outputs should be avg pooled for this to actually make sense
        pooled_embeddings=torch.nn.functional.adaptive_avg_pool2d(outs['conv'], (1, 1))
        conv_embeddings[batch_start:min(batch_start+batch_size, len(y)),:]=pooled_embeddings[:,:,0,0].cpu().detach().numpy()
        embeddings[batch_start:min(batch_start+batch_size, len(y)),:]=outs['avgpool'][:,:,0,0].cpu().detach().numpy()

        # computing prediction loss
        loss = torch.nn.CrossEntropyLoss()
        pred = outs['fc']
        len_=pred.shape[0]
        target=np.zeros((len_, 1000))
        for i in range(len(pred)):
            target[i,int(batch_y[i])]=1.
        target=torch.tensor(target, requires_grad=True).to(device)
        outloss = loss(pred, target)

        # Storing predictions
        softmaxf = torch.nn.Softmax(dim=1)
        predictions[batch_start:min(batch_start+batch_size, len(y)),:]=softmaxf(pred).detach().cpu()

        # Computing the gradients and storing them 
        grads_wrt_conv = torch.autograd.grad(outloss, outs['conv'], retain_graph=True)[0]
        gradients_wrt_conv_layer[batch_start:min(batch_start+batch_size, len(y)),:,:,:] = grads_wrt_conv[:,:,:,:].cpu()
        conv_maps[batch_start:min(batch_start+batch_size, len(y)),:,:,:] = outs['conv'].cpu().detach()

        grads = torch.autograd.grad(outloss, outs['avgpool'], retain_graph=True)[0]
        gradients[batch_start:min(batch_start+batch_size, len(y)),:] = grads[:,:,0,0].cpu()

        batch_start += batch_size

    print(f"gradients shape {gradients.shape}, conv_embs shape {conv_embeddings.shape}, conv_maps.shape {conv_maps.shape}")
    """
    SAVE INTERMEDIATE RESULTS
    """
    np.save(f"{SAVEFOLD}/predictions.npy", predictions)
    np.save(f"{SAVEFOLD}/gradients_wrt_conv_layer.npy", gradients_wrt_conv_layer)
    np.save(f"{SAVEFOLD}/conv_maps.npy", conv_maps)

