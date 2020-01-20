# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import json
import helper
import argparse


parser = argparse.ArgumentParser(
    description='Load and run model to classify flowers. Using resnet34.',
)
parser.add_argument('Path_to_image_file', action='store', type=str, help='path to flower image')
parser.add_argument('Path_to_saved_model_checkpoint', action='store', type=str, help='path to model checkpoint')
args = parser.parse_args()


# Loading model
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    model = models.resnet34(pretrained=True)
    # model = models.densenet169(pretrained=True) # Uncomment based on the pre-trained network used

    for param in model.parameters():
        param.requires_grad = False

    # Core model functions for resnet34
    model.fc = nn.Sequential(nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 256),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(256, 256),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(256, 102),
                             nn.LogSoftmax(dim=1))

    # # Core model functions densenet169
    # model.classifier = nn.Sequential(nn.Linear(1664, 832),
    #                                  nn.ReLU(),
    #                                  nn.Dropout(0.2),
    #                                  nn.Linear(832, 416),
    #                                  nn.ReLU(),
    #                                  nn.Dropout(0.2),
    #                                  nn.Linear(416, 204),
    #                                  nn.ReLU(),
    #                                  nn.Dropout(0.2),
    #                                  nn.Linear(204, 102),
    #                                  nn.LogSoftmax(dim=1))

    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['state_dict'])
    return model


model = load_checkpoint(args.Path_to_saved_model_checkpoint)
# model = load_checkpoint(args.Path_to_saved_model_checkpoint) # for densenet169

# Predict name of flower
probs, classes = helper.predict(args.Path_to_image_file, model)

# Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Print results
helper.print_prediction_results(args.Path_to_image_file, probs, classes, cat_to_name)
