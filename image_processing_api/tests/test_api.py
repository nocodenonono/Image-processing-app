"""
test api working properly
"""
import pytest
from image_processing_api import image_api

app = image_api.app

headers = {
    'Content-Type': 'multipart/form-data'
}


def get_data():
    """
    load image data and return as dictionary for post data
    :return: image data
    """
    image_fn = 'image_processing_api/tests/test.jpg'
    image = open(image_fn, 'rb')
    return {
        'image': (image, image_fn),
    }


@pytest.fixture(scope='session')
def client():
    """
    initialize test client
    """
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


def test_blur_request(client):
    """
    test can correctly post an image on the blur api
    """
    rv = client.post('/blur?sigma=3', headers=headers, data=get_data())

    assert rv.status_code == 200


def test_scale_request(client):
    """
    test can correctly post an image on the scale api
    """
    rv = client.post('/scale?factor=2', headers=headers, data=get_data())

    assert rv.status_code == 200


def test_cartoon_request(client):
    """
    test can correctly post an image on the cartoon api
    """
    rv = client.post('/cartoon', headers=headers, data=get_data())

    assert rv.status_code == 200


def test_seam_carving_request(client):
    """
    test can correctly post an image on the seam api
    """
    rv = client.post('seam_carving?reduced_size=1', headers=headers, data=get_data())

    assert rv.status_code == 200


def test_contrast_enhancement(client):
    """
    test contrast enhancement route
    """
    rv = client.post('contrast?clip=40', headers=headers, data=get_data())

    assert rv.status_code == 200


def test_histogram_equalization(client):
    """
    test histogram equalization route
    """
    rv = client.post('histogram', headers=headers, data=get_data())

    assert rv.status_code == 200


def test_histogram_visualization(client):
    """
    test histogram visualization route
    """
    rv = client.post('histogram_vis', headers=headers, data=get_data())

    assert rv.status_code == 200
