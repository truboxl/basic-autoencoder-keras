#!/usr/bin/python3

# Import modules
import math
import numpy as np

from PIL import Image

from tensorflow.image import ssim, rgb_to_yuv
from tensorflow import convert_to_tensor

# DEPRECATED: skimage.measure.compare_ssim has been moved to skimage.metrics.structural_similarity
from skimage.metrics import structural_similarity as compare_ssim
import cv2

from scipy import signal
from scipy.ndimage.filters import convolve

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

# Load decoded image
path_decoded = 'image_decoded.jpg'

decoded_image = []

print(f'Opening {path_decoded}')
x = Image.open(path_decoded)
x = x.convert('RGB') # for grayscale or RGBA
print('Decoded image size is ' + str(x.size))
x = make_square(x)
print('Squared image size is ' + str(x.size))
x = x.resize(target_image_size) # TAKE NOTE
x = np.asarray(x)
x = x.astype('float32')/255
decoded_image.append(np.asarray(x)) # can take a long time

#print(len(decoded_image))
#print(decoded_image[0].shape)

decoded_image = np.reshape(decoded_image, (len(decoded_image), target_length_size, target_length_size, 3))
print('decoded_image type is ' + str(type(decoded_image)))
print('decoded_image shape is '+ str(np.asarray(decoded_image).shape)) # type list has no shape, must be in type array # (1, 128, 128, 3)
print('image type in decoded_image is ' + str(type(decoded_image[0])))

# Single image from array
i = 0

inputs = original_image
decoded = decoded_image

# CALCULATION STARTS HERE

# https://dsp.stackexchange.com/questions/38065/peak-signal-to-noise-ratio-psnr-in-python-for-an-image

def mse(img1, img2):
    return np.mean( (img1 - img2) ** 2 )

def psnr(img1, img2):
    mserr = mse(img1, img2)
    PIXEL_MAX = 1.0
    try:
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mserr))
    except ZeroDivisionError:
        return 'Same image'

n = 1
for i in range(n):
    original = inputs[i]
    contrast = decoded[i]
    a = mse(original, contrast)
    b = psnr(original, contrast) # can also use cv2.PSNR rather than defined function psnr
    if b != 'Same image':
        print(f'MSE = {a:.4f}, PSNR (dB) = {b:.2f}')
    else:
        print(f'MSE = {a:.4f}, PSNR too high')

# Tensorflow implementation of SSIM

original = inputs[i].reshape(1, 128, 128, 3)
original = convert_to_tensor(original) # to use tf functions
original = rgb_to_yuv(original) # ssim only accept YUV, Grayscale only

contrast = decoded[i].reshape(1, 128, 128, 3)
contrast = convert_to_tensor(contrast)
contrast = rgb_to_yuv(contrast)

a = np.asarray(ssim(original, contrast, max_val=1)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
print(f'SSIM (TF) = {a[0]:.4f}')

# Scikit Image implementation of SSIM

grayA = cv2.cvtColor(inputs[0], cv2.COLOR_RGB2GRAY)
grayB = cv2.cvtColor(decoded[0], cv2.COLOR_RGB2GRAY)
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print(f'SSIM (SCK) = {score:.4f}')

# MSSIM
# https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py

def _FSpecialGauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  radius = size // 2
  offset = 0.0
  start, stop = -radius, radius + 1
  if size % 2 == 0:
    offset = 0.5
    stop -= 1
  x, y = np.mgrid[offset + start:stop, offset + start:stop]
  assert len(x) == size
  g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
  return g / g.sum()


def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
  """Return the Structural Similarity Map between `img1` and `img2`.
  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  _, height, width, _ = img1.shape

  # Filter size can't be larger than height or width of images.
  size = min(filter_size, height, width)

  # Scale down sigma if a smaller filter size is used.
  sigma = size * filter_sigma / filter_size if filter_size else 0

  if filter_size:
    window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
    sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
  else:
    # Empty blur kernel so no need to convolve.
    mu1, mu2 = img1, img2
    sigma11 = img1 * img1
    sigma22 = img2 * img2
    sigma12 = img1 * img2

  mu11 = mu1 * mu1
  mu22 = mu2 * mu2
  mu12 = mu1 * mu2
  sigma11 -= mu11
  sigma22 -= mu22
  sigma12 -= mu12

  # Calculate intermediate values used by both ssim and cs_map.
  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  v1 = 2.0 * sigma12 + c2
  v2 = sigma11 + sigma22 + c2
  ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
  cs = np.mean(v1 / v2)
  return ssim, cs


def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5,
                   k1=0.01, k2=0.03, weights=None):
  """Return the MS-SSIM score between `img1` and `img2`.
  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.
  Returns:
    MS-SSIM score between `img1` and `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
  if img1.shape != img2.shape:
    raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
  if img1.ndim != 4:
    raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

  # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
  weights = np.array(weights if weights else
                     [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
  levels = weights.size
  downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
  im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
  mssim = np.array([])
  mcs = np.array([])
  for _ in range(levels):
    ssim, cs = _SSIMForMultiScale(
        im1, im2, max_val=max_val, filter_size=filter_size,
        filter_sigma=filter_sigma, k1=k1, k2=k2)
    mssim = np.append(mssim, ssim)
    mcs = np.append(mcs, cs)
    filtered = [convolve(im, downsample_filter, mode='reflect')
                for im in [im1, im2]]
    im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
  return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) *
          (mssim[levels-1] ** weights[levels-1]))

original = inputs[i].reshape(1, 128, 128, 3)
contrast = decoded[i].reshape(1, 128, 128, 3)
d = MultiScaleSSIM(original, contrast, max_val=1)
print(f'MSSIM = {d:.4f}')
