# Imports here
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import json
import model_architecture

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Load data from user/script
parser = argparse.ArgumentParser(
    description='Training image classifier neural network using resnet34 architecture.',
)
parser.add_argument('Path_to_dataset', action='store', type=str, help='path to dataset')

parser.add_argument('--learning_rate', dest='learning_rate', type=float, action='store', default=0.003,
                    help='learning rate for training')

parser.add_argument('--hidden_units', dest='hidden_units', type=int, nargs='+', action='store',
                    help='add a list of hidden layer sizes')

parser.add_argument('--epochs', dest='number_of_epochs', type=int, action='store', default=10,
                    help='number of epochs for training')

parser.add_argument('--gpu', dest='use_gpu', action='store_true',
                    help='use gpu if available')

parser.add_argument('--arch', dest='arch', action='store', default='resnet34',
                    help='models supported densenet169 and resnet34')

parser.add_argument('--save_dir', dest='Path_to_checkpoint', action='store', default='checkpoint.pth',
                    help='path to model checkpoint')

args = parser.parse_args()

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Load the data
"""Notes: The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was 
normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to 
what the network expects. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 
0.225], calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range 
from -1 to 1. """

data_dir = args.Path_to_dataset
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_data_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_valid_data_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_data_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_valid_data_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_data_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Loading pretrained model

"""
For this project, the model has been trained on two different neural networks: densenet169 and resnet34. 
Please ensure to comment/uncomment code based on the neural network.
"""
model = models.densenet169(pretrained=True) if args.arch =='densenet169' else models.resnet34(pretrained=True)
print("------------------------------")
print("Model Arch Selected: ", args.arch)
print("------------------------------")
print("Printing pretrained model's parameters", model)
print("------------------------------")

# Turn of gradients for model features
for param in model.parameters():
    param.requires_grad = False

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Defining classifier

device = torch.device("cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
print('Model is using: ', device)

input_size = 512

if args.arch == 'densenet169':
    input_size = model.classifier.in_features
    model.classifier = model_architecture.ModelArch('densenet169', input_size, 102, args.hidden_units)
elif args.arch == 'resnet34':
    input_size = model.fc.in_features
    model.fc = model_architecture.ModelArch('resnet34', input_size, 102, args.hidden_units)

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate) if args.arch =='densenet169' else optim.Adam(model.fc.parameters(), lr=args.learning_rate)
criterion = nn.NLLLoss()

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Moving model to device
model.to(device)

# Model hyperparameters
epochs = args.number_of_epochs
steps = 0
print_every = 5

# Training Model
train_losses, validation_loss = [], []
for epoch in range(epochs):
    running_loss = 0
    for features, labels in train_data_loader:
        steps += 1

        # Move data and labels to the active device
        features, labels = features.to(device), labels.to(device)

        # Resetting gradients
        optimizer.zero_grad()

        # Calculate output
        logps = model.forward(features)

        # Calculate loss
        loss = criterion(logps, labels)

        # Calculate gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    else:
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0

            with torch.no_grad():

                # Set model to evaluation mode
                model.eval()

                for features, labels in valid_data_loader:
                    # Move data and labels to the active device
                    features, labels = features.to(device), labels.to(device)

                    # Calculate output
                    logps = model.forward(features)

                    # Calculate loss
                    test_loss += criterion(logps, labels)

                    # Output as probabilities instead of log probabilities
                    porb_output = torch.exp(logps)

                    # Finding the class with the top probability
                    top_p, top_class = porb_output.topk(1, dim=1)

                    # Calculate matches between model output and labels and storing in equals. Note shapes are not same.
                    equals = top_class == labels.view(*top_class.shape)

                    # Calculate accuracy by taking % of times we get the right prediction.
                    accuracy += torch.sum(equals.type(torch.FloatTensor)) / equals.shape[0]

            # set model back to train mode
            model.train()

            train_losses.append(running_loss / len(train_data_loader))
            validation_loss.append(test_loss / len(valid_data_loader))

            print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_data_loader)),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(valid_data_loader)),
                  "Test Accuracy: {:.3f}".format(accuracy / len(valid_data_loader)))

# # Plotting test and training data loss uncomment for use
# plt.plot(train_losses, label='Training loss')
# plt.plot(validation_loss, label='Validation loss')
# plt.legend(frameon=False)
# plt.show()

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Testing network

accuracy = 0

# Set model to evaluation mode
model.eval()

with torch.no_grad():
    for images, labels in test_data_loader:
        # Move data and labels to the active device
        images, labels = images.to(device), labels.to(device)
        model.to(device)

        # Calculate output
        logps = model.forward(images)

        # Output as probabilities instead of log probabilities
        prob_output = torch.exp(logps)

        # Finding the class with the top probability
        top_p, top_class = prob_output.topk(1, dim=1)

        # Calculate matches between model output and labels and storing in equals.
        equals = top_class == labels.view(*top_class.shape)

        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += torch.sum(equals.type(torch.FloatTensor)) / equals.shape[0]

    print("Test Accuracy: {:.3f}".format(accuracy / len(test_data_loader)))

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Saving model
# TODO: Save the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': input_size,
              'output_size': 102,
              'hidden_layers': args.hidden_units,
              'state_dict': model.state_dict(),
              'class_mapping': model.class_to_idx,
              'model_arch': args.arch
              }

torch.save(checkpoint, args.Path_to_checkpoint)
