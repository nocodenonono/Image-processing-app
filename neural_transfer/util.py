import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


def image_loader(image_name, device):
    """
    Reference: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    Load an image from disk and transform it to tensor as network input
    """
    if isinstance(image_name, str):
        image = Image.open(image_name)
    else:
        image = image_name

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def show_image(image):
    """
    turns an image tensor to PIL image and plt it
    :param image: tensor, 1x3x256x256
    :return: None
    """
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)


def normalizer(device):
    """
    normalizer for input to VGG16 network
    :param device: cpu or cuda
    :return: normalizing function
    """
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    return transforms.Normalize(normalization_mean, normalization_std)


def gram(feature_map):
    """
    calculates the gram matrix of the feature map
    G(y) = y * y.t()

    :param feature_map: feature map output of some layer of neural network
    :return: gram matrix of the input
    """
    batch_size, nmaps, h, w = feature_map.size()
    features = feature_map.view(batch_size, nmaps, h * w)

    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(batch_size * nmaps * h * w)


def image_unloader(image_tensor):
    """
    unload image tensor to RGB
    :param image_tensor: tensor
    :return: RGB image
    """
    image_tensor = image_tensor.squeeze(0)
    image = unloader(image_tensor)
    return image





