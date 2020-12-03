"""
Test image processing algorithms
"""
import cv2
import os
from image_processing import basic_processing

image = cv2.imread('image_processing/tests/test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
H, W, _ = image.shape


def test_scale():
    """
    test scale down and up and incorrect factor
    """
    img = basic_processing.scale(image, 2)
    h, w, _ = img.shape

    assert h == 2 * H
    assert w == 2 * W

    img = basic_processing.scale(image, 0.5)
    h, w, _ = img.shape

    assert h == int(0.5 * H)
    assert w == int(0.5 * W)

    try:
        basic_processing.scale(image, 0)
        assert 0 == 1
    except ValueError:
        assert 1 == 1


def test_seam_carving():
    """
    test seam carving can reduce the width
    """
    img = basic_processing.seam_carving(image, 1)
    h, w, _ = img.shape

    assert h == H
    assert w == W - 1


def test_contrast_enhancement():
    """
    test contrast enhancement only changes
    """
    img = basic_processing.contrast_enhancement(image)

    assert img.shape == image.shape


def test_histogram_equalization():
    """
    test histogram_equalization
    """
    img = basic_processing.histogram_equalization(image)

    assert img.shape == image.shape


def test_histogram_visualization():
    """
    test histogram_visualization can save
    plot to file
    """
    basic_processing.histogram_visualization(image)

    assert os.path.exists('plot.png')
