# noiseremoval / fuckartifacts
Examples of different approaches of doing noise removal with neural networks. Primary intent is imagery but this applies to ever single modality of signal. 

## Installing this shit
> git clone yadayada....; cd yadayada.... # then make the venv:

> python3 -m venv thisshit

> ./thisshit/bin/pip install -r requirements.txt

> ./thisshit/bin/activate

By this time you should have a good setup for playing with this shit without fucking up your base py setup.


Then to quickly test your setup is working properly:

to test pytorch:
> python3 test-pytorch-setup.py

to test tensorflow:
> python3 test-tensorflow-setup.py


## Thingses
Then we have a couple of ways to denoise:


**Autoencoders** (tensorflow): Getting there. Some progress has been done.


**Generative Adversarial Networks** (GANs) (pytorch)


**Noise2Noise** (pytorch): Kinda works (Expects noisy stuff in ../data/training and will output to ../data/cleaned)


**CycleGAN** (tensorflow): Kinda works (Expects noisy stuff in ../data/training outputs to ../data/clean)


**Convolutional Neural Networks** (CNNs) (pytorch)



## Caveats:
Note that this is just example code and needs a lot of adaptation for them to work on your datasets.

For denoising without clean images to train on I would use noise2noise or cycleGAN.


And again: This is just boiler plate example code and does not work "out of the box" apart from some of the examples that will generate random data and train on it.

