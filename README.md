# noiseremoval / fuckartifacts
Examples of different approaches of doing noise removal with neural networks. Primary intent is imagery but this applies to ever single modality of signal. 

## Installing this shit
> pip install -r requirements.txt

Then to quickly test your setup is working properly:

to test pytorch:
> python3 test-pytorch-setup.py

to test tensorflow:
> python3 test-tensorflow-setup.py


## Thingses
Then we have a couple of ways to denoise:


**Autoencoders** (tensorflow)


**Generative Adversarial Networks** (GANs) (pytorch)


**Noise2Noise** (pytorch): Kinda works (Expects noise in ../data and will output to ../cleaned)


**CycleGAN** (tensorflow)


**Convolutional Neural Networks** (CNNs) (pytorch)



## Caveats:
Note that this is just example code and needs a lot of adaptation for them to work on your datasets.

For denoising without clean images to train on I would use noise2noise or cycleGAN.


And again: This is just boiler plate example code and does not work "out of the box" apart from some of the examples that will generate random data and train on it.

