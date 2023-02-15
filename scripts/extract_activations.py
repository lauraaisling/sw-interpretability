# python scripts/extract_activations.py --model 'googlenet_torchhub' --layer inception4e.branch1.conv --in_folder data/test-images/test1 --out_folder results/test1 --save_file googlenet_torchhub_test
# python scripts/extract_activations.py --model 'googlenet_captum' --layer mixed4e.conv_1x1 --in_folder data/test-images/test1 --out_folder results/test1 --save_file googlenet_captum_test

# python scripts/extract_activations.py --model 'googlenet_torchhub' --layer inception4e.branch1.conv --in_folder data/test-images/OpenAI_microscope/car --out_folder results/OpenAI_microscope_acts/inception4e_branch1_conv --save_file openai_car
# python scripts/extract_activations.py --model 'googlenet_captum' --layer mixed4e.conv_1x1 --in_folder data/test-images/OpenAI_microscope/car --out_folder results/OpenAI_microscope_acts/mixed4e_conv_1x1 --save_file openai_car

import argparse
import sys
import os
import utilities
import numpy as np

base_dir = os.path.normpath(os.getcwd())
sys.path.append(f"{base_dir}/src/captum")
from captum import optim as optimviz
import torch
import torchvision
from PIL import Image
import json
from torchvision.models import feature_extraction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Models:
# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
# GoogleNet:
# https://github.com/pytorch/captum/blob/optim-wip/captum/optim/models/_image/inception_v1.py line 11
# torchvision googlenet:
# https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
def get_model(model_name):
    if model_name == "googlenet_torchhub":
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "googlenet", weights=torchvision.models.GoogLeNet_Weights.DEFAULT
        ).to(device)
        model.eval()

    # elif model_name == "googlenet_torchvision":
    #     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     model = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1).to(device)
    #     model.eval()

    elif model_name == "googlenet_captum": # Nope!!! 
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = optimviz.models._image.inception_v1.googlenet(pretrained=True, replace_relus_with_redirectedrelu = False).to( 
            device
        )
        model.eval()

    return model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model)

    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    ##################################################################
    ### begin method 2
    # return_nodes = {}
    # return_nodes[args.layer] = args.layer 
    # model_extr = feature_extraction.create_feature_extractor(model, return_nodes=return_nodes)
    # #oldcode# for i in range(len(args.layer)):
    # #oldcode#    return_nodes[args.layer[i]] = args.layer[i]
    # #oldcode# for module, layer_name in model.named_modules():
    # #oldcode#     return_nodes[module] = module
    # #oldcode# del return_nodes['']
    ### end method 2
    ##################################################################

    embeddings = {}
    batch_start = 0
    batch_size = 3#2 
    folder = args.in_folder
    while batch_start + batch_size < len(os.listdir(folder)) + batch_size:
        if args.model == "googlenet_captum":
            inputs = torch.stack([
                utilities.preprocess_captum(Image.open(f'{folder}/{os.listdir(folder)[i]}').convert("RGB"))
                for i in range(batch_start,min(batch_start + batch_size,len(os.listdir(folder))))
            ])
        else: 
            inputs = torch.stack([
                # utilities.preprocess(
                Image.open(f'{folder}/{os.listdir(folder)[i]}').convert("RGB")#)
                for i in range(batch_start,min(batch_start + batch_size,len(os.listdir(folder))))
            ])
        # print(inputs) # TODO captum models pp
        # print(inputs.shape)
        inputs = inputs.clone().detach().requires_grad_(True)
        inputs = inputs.to(device)
        #################################################################
        ### begin method 1
        # https://discuss.pytorch.org/t/how-to-register-forward-hooks-for-each-module/43347
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254
        # https://discuss.pytorch.org/t/how-to-save-every-visualization-of-conv2d-activation-layer/78620
        activation = {}
        def get_activation(layer): # , model
            def hook(model, input, output):
                activation[layer] = output.detach()
            return hook 
        
        for module, layer_name in model.named_modules():
            if module == args.layer:
                layer_name.register_forward_hook(get_activation(module)) # , model
        with torch.no_grad():
            outs = model(inputs)
        for module, layer_name in model.named_modules():
            if module == args.layer:
                if batch_start == 0:
                    embeddings[module] = activation[args.layer] #.detach()
                else:
                    embeddings[module] = torch.cat((embeddings[module], activation[args.layer])) # .detach()
        ### end method 1
        ##################################################################
        ### begin method 2
        # outs = model_extr(inputs) 
        # outs[args.layer].cpu().detach().numpy() 
        # for module in return_nodes:
        #     if batch_start == 0:
        #         embeddings[module] = outs[module].detach()
        #     else:
        #         embeddings[module] = torch.cat((embeddings[module], outs[module].detach())) 
        ### end method 2
        ##################################################################
        batch_start+=batch_size 
    if not os.path.exists(f'/{args.out_folder}'):
        os.makedirs(f'{args.out_folder}', exist_ok=True)
    np.save(f'{args.out_folder}/{args.save_file}.npy', embeddings[args.layer].cpu().detach().numpy())
    # print(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute activations")
    parser.add_argument("--model", help="the model version")
    parser.add_argument("--layer", help="layer name")
    parser.add_argument("--in_folder", help="the input folder")
    parser.add_argument("--out_folder", help="the output folder")
    parser.add_argument("--save_file", help="the output file name")
    args = parser.parse_args()

    main(args)
