#!/usr/bin/python3

# Import modules
import matplotlib.pyplot as plt
import pickle as pkl

from tensorflow.keras.models import load_model

# Load Model
decoder = load_model('image_decoder_model.h5')
decoder.summary()

target_length_size = 128
target_image_size = (target_length_size, target_length_size)

# Load encoded image

with open('image_encoded.pkl', 'rb') as handle:
    print('Loading image_encoded.pkl')
    encoded = pkl.load(handle)
    print('Done')

# Predict
print('Running decoding')
decoded_imgs = decoder.predict(encoded)
decoded = decoded_imgs

# Single image from array
i = 0

# Save image decoded
print('Saving image_decoded.jpg')
plt.imsave('image_decoded.jpg', decoded[i])
