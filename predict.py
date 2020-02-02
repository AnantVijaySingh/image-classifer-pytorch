# Imports here
import torch
from torchvision import models
import json
import helper
import argparse
import model_architecture

parser = argparse.ArgumentParser(
    description='Load and run model to classify flower images.',
)
parser.add_argument('Path_to_image_file', action='store', type=str, help='path to flower image')

parser.add_argument('Path_to_saved_model_checkpoint', action='store', type=str, help='path to model checkpoint')

parser.add_argument('--top_k', dest='top_k_classes', type=int, action='store', default=5,
                    help='number of top K most likely classes to be returned')

parser.add_argument('--category_names', dest='category_names_json', action='store', default='cat_to_name.json',
                    help='number of top K most likely classes to be returned')

parser.add_argument('--gpu', dest='use_gpu', action='store_true',
                    help='use gpu if available')

args = parser.parse_args()

device = torch.device("cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
print('Model is using: ', device)


# Loading model
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet169(pretrained=True) if checkpoint['model_arch'] == 'densenet169' else models.resnet34(
        pretrained=True)

    # Turn of gradients for model features
    for param in model.parameters():
        param.requires_grad = False

    if checkpoint['model_arch'] == 'densenet169':
        model.classifier = model_architecture.ModelArch('densenet169', checkpoint['input_size'],
                                                        checkpoint['output_size'], checkpoint['hidden_layers'])
    elif checkpoint['model_arch'] == 'resnet34':
        model.fc = model_architecture.ModelArch('resnet34', checkpoint['input_size'], checkpoint['output_size'],
                                                checkpoint['hidden_layers'])

    model.load_state_dict(checkpoint['state_dict'])

    return model


model = load_checkpoint(args.Path_to_saved_model_checkpoint)

# Predict name of flower
probs, classes = helper.predict(args.Path_to_image_file, model, device, args.top_k_classes)

# Mapping
with open(args.category_names_json, 'r') as f:
    cat_to_name = json.load(f)

# Print results
helper.print_prediction_results(args.Path_to_image_file, probs, classes, cat_to_name)
