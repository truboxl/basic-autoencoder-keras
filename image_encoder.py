#!/usr/bin/python3

# Import modules
import numpy as np
import pickle as pkl

from PIL import Image
from tensorflow.keras.models import load_model

# Load Model
print('Loading image_encoder_model.h5')
encoder = load_model('image_encoder_model.h5')
encoder.summary()

# Preprocess
def make_square(img):
    cols,rows = img.size
    
    if rows>cols:
        pad = (rows-cols)/2
        img = img.crop((0, pad, cols, cols+pad)) # left, up, right, down
    else:
        pad = (cols-rows)/2
        img = img.crop((pad, 0, rows+pad, rows))
    
    return img # make centered square image

target_length_size = 128
target_image_size = (target_length_size, target_length_size)

# Load Image
path_image = 'image.jpg'

original_image = []

print(f'Opening {path_image}')
x = Image.open(path_image)
x = x.convert('RGB') # for grayscale or RGBA
print('Original image size is ' + str(x.size))
x = make_square(x)
print('Squared image size is ' + str(x.size))
x = x.resize(target_image_size) # TAKE NOTE
x = np.asarray(x)
x = x.astype('float32')/255
original_image.append(np.asarray(x)) # can take a long time

#print(len(original_image))
#print(original_image[0].shape)

original_image = np.reshape(original_image, (len(original_image), target_length_size, target_length_size, 3))
print('original_image type is ' + str(type(original_image)))
print('original_image shape is '+ str(np.asarray(original_image).shape)) # type list has no shape, must be in type array # (1, 128, 128, 3)
print('image type in original_image is ' + str(type(original_image[0])))

# Predict
print('Running encoding')
encoded_imgs = encoder.predict(original_image)

# image number from dataset
i = 0

inputs = original_image
encoded = encoded_imgs

with open('image_encoded.pkl', 'wb') as handle:
    print('Saving image_encoded.pkl')
    pkl.dump(encoded, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print('Done')
