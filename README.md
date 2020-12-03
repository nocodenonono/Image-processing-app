# A mobile app that implements some image processing algorithms

## Structure

- React Native front end
- Flask backend

## Basic algorithms

- blur
- scale
- content-aware resizing
- cartoonize
- histogram equalization
- CLAHE contrast enhancement
- Visualization of pixel values using bar chart

## advanced algorithms

- neural style transfer

## Additional support functionality

- supports save processed image to local album
- revert to original image from processed image

## Limitation
- All processing happens in backend not locally
- does not support multiple processing (i.e. you cannot blur and resize the image at the same time,
all the processing happens to the original image)
However, you can save the processed image to album and then process that image.