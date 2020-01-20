from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)

    # Provide the target width and height of the image
    req_width = 256
    width_scaling = (req_width / float(image.size[0]))
    scaled_height = int((float(image.size[1]) * float(width_scaling)))

    resized_image = image.resize((req_width, scaled_height), Image.ANTIALIAS)

    # Cropping image
    width, height = resized_image.size  # Get dimensions
    crop_width, crop_height = (224, 224)

    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2

    # Crop the center of the image
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Converting color channel values to floats
    np_image = np.array(cropped_image, dtype=float)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image_converted = np_image / 255

    # Normalizing image
    np_image_normalized = (np_image_converted - mean) / std

    # Transpose
    np_image_transposed = np_image_normalized.transpose((2, 0, 1))

    image_tensor = torch.from_numpy(np_image_transposed)

    return image_tensor


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.to('cpu')
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.show()

    return ax


def predict(image_path, model, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """

    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        # Set model to evaluation mode
        model.eval()

        # Move data and labels to the active device
        processed_image = processed_image.to(device)

        # Calculate output
        logps = model.forward(processed_image.unsqueeze_(0).float())

        # Output as probabilities instead of log probabilities
        prob_output = torch.exp(logps)

        # Finding the class with the top probability
        top_p, top_class = prob_output.topk(5, dim=1)

        print(top_p)
        print(top_class)

        return top_p, top_class


# TODO: Display an image along with the top 5 classes
def view_classify(image_path, img_probs, img_classes):
    """
    Function for viewing an image and it's predicted classes.
    """
    img = process_image(image_path)
    img_probs = img_probs.to('cpu')
    img_classes = img_classes.to('cpu')

    probs = img_probs.data.numpy().squeeze()
    classes = img_classes.numpy().squeeze()
    imshow(img)

    class_names = np.array([
        cat_to_name[str(classes[0])],
        cat_to_name[str(classes[1])],
        cat_to_name[str(classes[2])],
        cat_to_name[str(classes[3])],
        cat_to_name[str(classes[4])],
    ])

    plt.barh(class_names, probs)
    plt.show()


# TODO: Display an image along with the top 5 classes
def print_prediction_results(image_path, img_probs, img_classes, mapping):
    """
    Function for viewing an image and it's predicted classes.
    """
    img = process_image(image_path)
    img_probs = img_probs.to('cpu')
    img_classes = img_classes.to('cpu')

    probs = img_probs.data.numpy().squeeze()
    classes = img_classes.numpy().squeeze()

    for i in range(5):
        print("Predicted Flower Name: {:20s} ".format(mapping[str(classes[i])]),
              "Probability: {:.3f}.. ".format(probs[i]))
