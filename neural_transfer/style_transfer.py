import neural_transfer.model.transform_net as net
from neural_transfer.util import *


def style_transfer(style_model_pth, image_fn):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.ImageTransformNet()
    model.load_state_dict(torch.load(style_model_pth, map_location=torch.device(device)))
    model.eval()

    image_tensor = image_loader(image_fn, device)
    output = model(image_tensor)
    return output


if __name__ == '__main__':
    style_transfer('network/mosaic.pth', 'test.jpg')
