# noiseremoval
Examples of different approaches of doing noise removal with neural networks.

pip install -r requirements.txt

Then to quickly test your setup is working properly:
python3 test-pytorch-setup.py
python3 test-tensorflow-setup.py

Then we have a couple of ways to denoise:

Autoencoders (tensorflow)
Generative Adversarial Networks (GANs) (pytorch)
Noise2Noise (pytorch)
CycleGAN (tensorflow)
Convolutional Neural Networks (CNNs) (pytorch)

Note that this is just example code and needs a lot of adaptation for them to work on your datasets.
For denoising without clean images to train on I would use noise2noise or cycleGAN.

And again: This is just boiler plate example code and does not work "out of the box" apart from some of the examples that will generate random data and train on it.

