# python scripts/extract_activations.py --model 'googlenet_torchhub' --layer 'inception4e.branch1.conv' --in_folder data/test-images/cat --out_folder results --save_file cat
import argparse
import sys
import os
import utilities

base_dir = os.path.normpath(os.getcwd())
sys.path.append(f"{base_dir}/src/captum")
from captum import optim as optimviz
import torch
import torchvision
from PIL import Image
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Models:
# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
# GoogleNet:
# https://github.com/pytorch/captum/blob/optim-wip/captum/optim/models/_image/inception_v1.py line 11
# torchvision googlenet:
# https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
def get_model(model_name):
    if model_name == "googlenet_torchhub":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "googlenet", pretrained=True
        ).to(device)
        model.eval()

    elif model_name == "googlenet_torchvision":
        model = torchvision.models.googlenet(pretrained=True).to(device)
        model.eval()

    elif model_name == "googlenet_captum":
        model = optimviz.models._image.inception_v1.googlenet(pretrained=True).to(
            device
        )
        model.eval()

    return model


# https://discuss.pytorch.org/t/how-to-register-forward-hooks-for-each-module/43347
# https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254
# https://discuss.pytorch.org/t/how-to-save-every-visualization-of-conv2d-activation-layer/78620
activation = {}


def get_activation(layer, model):
    # activation = {}
    def hook(model, input, output):
        activation[layer] = output.detach()

    return hook


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model)

    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    # activations_dict = {}
    activations = torch.empty(0, 256, 14, 14)  ########################
    for file in os.listdir(args.in_folder):
        # activation = {}
        input_image = Image.open(args.in_folder + "/" + file).convert(
            "RGB"
        )  # since 'RGBA'
        input_tensor = utilities.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        for module, layers in model.named_modules():
            if module == args.layer:
                layers.register_forward_hook(get_activation(module, model))
        with torch.no_grad():
            output = model(input_batch)
        activations = torch.cat((activations, activation[args.layer]))
    print(activations)
    torch.save(activations, f"{args.out_folder}/acts_{args.save_file}_{args.layer}.pt")

    # activations_dict[args.layer] = activations
    # activations_json = json.dumps(activations_dict)

    # with open(f'{args.out_folder}/acts_{args.in_folder}.json','w') as file:
    #    file.write(activations_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute activations")
    parser.add_argument("--model", help="the model version")
    parser.add_argument("--layer", help="the layer name")
    parser.add_argument("--in_folder", help="the input folder")
    parser.add_argument("--out_folder", help="the output folder")
    parser.add_argument("--save_file", help="the output file name")
    args = parser.parse_args()

    main(args)
