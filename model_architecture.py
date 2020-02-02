from torch import nn


# Defining Model
class ModelArch(nn.Sequential):
    def __init__(self, arch, input_size, output_size=102, hidden_layers=None, drop_p=0.2):
        super().__init__()

        if hidden_layers is None and arch == 'resnet34':
            hidden_layers = [512, 256, 256]
        elif hidden_layers is None and arch == 'densenet169':
            hidden_layers = [832, 416, 204]

        # Input to hidden layer
        self.add_module("Input to Hidden", nn.Linear(input_size, hidden_layers[0]))
        self.add_module("ReLU", nn.ReLU())
        self.add_module("drop", nn.Dropout(drop_p))

        # Add a variable number of more hidden layers
        for h1, h2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.add_module("Hidden Layers {}".format(h1, h2), nn.Linear(h1, h2))
            self.add_module("ReLU", nn.ReLU())
            self.add_module("drop", nn.Dropout(drop_p))

        self.add_module("Hidden to Output", nn.Linear(hidden_layers[-1], output_size))
        self.add_module("Output", nn.LogSoftmax(dim=1))
