"""
API that do some image processing tasks
"""
from flask import Flask, request, send_file
from image_processing import basic_processing
from PIL import Image
from torchvision.utils import save_image
from neural_transfer.style_transfer import style_transfer as transfer
import numpy as np
import cv2

app = Flask(__name__)
app.config.from_mapping(SECRET_KEY='dev')


def send_image(image, name):
    """
    send the processed image as file
    :param image: image to send
    :param name: image file name
    :return: file sent
    """
    image = Image.fromarray(image.astype(np.uint8))
    image.save(name)
    return send_file(name)


def handle_input(var=None):
    """
    handles request params and data, return 400 if bad requests
    :param var: request params
    :return: parsed params
    """
    if var is not None:
        var = request.args.get(var)

    image = request.files['image']
    return image, var


@app.route('/scale', methods=['POST'])
def scale():
    """
    scale an image given factor
    :return: image scaled
    """
    try:
        image, factor = handle_input('factor')
    except KeyError:
        return 'Bad Request', 400

    image = np.array(Image.open(image.stream))
    try:
        image = basic_processing.scale(image, float(factor))
    except cv2.error:
        return 'Bad Request', 400
    return send_image(image, 'scaled.png')


@app.route('/blur', methods=['POST'])
def blur():
    """
    gaussian blur an image given standard deviation
    :return: blurred image
    """
    try:
        image, sigma = handle_input('sigma')
    except KeyError:
        return 'Bad Request', 400

    image = np.array(Image.open(image.stream))
    image = basic_processing.blur(image, int(sigma))
    return send_image(image, 'blurred.png')


@app.route('/cartoon', methods=['POST'])
def cartoonize():
    """
    cartoonize an image
    :return: cartoon image
    """
    try:
        image, _ = handle_input()
    except KeyError:
        return 'Bad Request', 400

    image = np.array(Image.open(image.stream))
    image = basic_processing.cartoonize(image)
    return send_image(image, 'cartoon.png')


@app.route('/seam_carving', methods=['POST'])
def seam_carving():
    """
    resize the image given number of seams to remove
    :return: resized image
    """
    try:
        image, num_seam = handle_input('reduced_size')
    except KeyError:
        return 'Bad Request', 400

    image = np.array(Image.open(image.stream))
    try:
        image = basic_processing.seam_carving(image, int(num_seam))
    except cv2.error:
        return 'Bad Request', 400
    return send_image(image, 'seam.png')


@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    """
    Using mosaic style on the input image
    :return: styled image
    """
    try:
        image, _ = handle_input()
    except KeyError:
        return 'Bad Request', 400

    image = Image.open(image.stream)
    image = transfer('../neural_transfer/network/mosaic.pth', image)
    save_image(image, 'style.png')
    return send_file('style.png')


@app.route('/contrast', methods=['POST'])
def contrast_enhancement():
    """
    enhance contrast by CLAHE
    :return: enhanced image file
    """
    try:
        image, clip = handle_input('clip')
    except KeyError:
        return 'Bad Request', 400

    image = np.array(Image.open(image.stream))
    image = basic_processing.contrast_enhancement(image, int(clip))
    return send_image(image, 'contrast.png')


@app.route('/histogram', methods=['POST'])
def histogram_equalization():
    """
    Histogram equalization
    :return: enhanced image file
    """
    try:
        image, _ = handle_input()
    except KeyError:
        return 'Bad Request', 400

    image = np.array(Image.open(image.stream))
    image = basic_processing.histogram_equalization(image)
    return send_image(image, 'hist_eq.png')


@app.route('/histogram_vis', methods=['POST'])
def histogram_visualization():
    """
    visualize image pixel values
    :return: histogram plot image file
    """
    try:
        image, _ = handle_input()
    except KeyError:
        return 'Bad Request', 400

    image = np.array(Image.open(image.stream))
    basic_processing.histogram_visualization(image)

    return send_file('plot.png')


if __name__ == '__main__':
    app.run()
