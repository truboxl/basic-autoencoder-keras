# fyp2020-basic-autoencoder-keras

Please refer [the presentation slide](./ea16062-fyp-presentation-slide.pdf)

This is an undergraduate final year project for year 2020 to investigate image compression using deep neural network and identify parameters that can optimise that use case.

This final year project uses COCO image dataset for training, Tensorflow and Keras to generate the neural networks, and PlaidML to accelerate neural network training using ordinary Intel GPUs.

## How to replicate

1. Obtain and preprocess the dataset images before training [#](https://github.com/truboxl/fyp2020-basic-autoencoder-keras/blob/master/Dataset%20Preprocess%20to%20Pickle%20-%20COCO%20train2017%20-%20128x128x3.ipynb)
2. Generate and train the encoder and decoder neural networks [#](https://github.com/truboxl/fyp2020-basic-autoencoder-keras/blob/master/Deep%20Autoencoders%20in%20Keras%20-%20conv-plaidml-v9-coco-make.ipynb)
3. Example using the neural networks [#](https://github.com/truboxl/fyp2020-basic-autoencoder-keras/blob/master/Deep%20Autoencoders%20in%20Keras%20-%20conv-plaidml-v9-coco-reuse.ipynb)
4. Deployment: [Encoder](https://github.com/truboxl/fyp2020-basic-autoencoder-keras/blob/master/image_encoder.py), [Decoder](https://github.com/truboxl/fyp2020-basic-autoencoder-keras/blob/master/image_decoder.py), [Difference calculator](https://github.com/truboxl/fyp2020-basic-autoencoder-keras/blob/master/image_calculate.py)
