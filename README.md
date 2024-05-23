# noiseremoval / fuckartifacts
Examples of different approaches of doing noise removal with neural networks. Primary intent is imagery but this applies to ever single modality of signal. 

## Installing this shit
Make sure you have cuda 12.3 and libcudnn-frontend-dev/mantic,mantic 0.8+ds-1 all + nvidia-cudnn/mantic 8.9.2.26~cuda12+1 amd64 ... maybe other things as well.. Not really sure..


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


**Autoencoders** (tensorflow): Kinda works.


**Noise2Noise** (pytorch): Kinda works.


**CycleGAN** (tensorflow): Kinda works.



## Caveats:
Note that this is just example code and needs a lot of adaptation for them to work on your datasets.
All of them expects input in data/training and will output in their own directory under data. 


