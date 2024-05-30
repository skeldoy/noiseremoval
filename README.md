# noiseremoval
Examples of different approaches of doing noise removal with neural networks. Primary intent is imagery but this applies to ever single modality of signal. 
The approaches are good for situations where you don't have any clean/denoised imagery to work with. Examples of that can be weird sensors, old photos, files with artifacts from compression and/or hardware. I have not gotten around to actually testing this stuff with a decent dataset, so I am not really sure how the results are. 

## Installing
Make sure you have cuda 12.3 and libcudnn-frontend-dev/mantic,mantic 0.8+ds-1 all + nvidia-cudnn/mantic 8.9.2.26~cuda12+1 amd64 ... maybe other things as well.. Not really sure..


> git clone yadayada....; cd yadayada.... # then make the venv:

> python3 -m venv .

> source ./bin/activate

> ./bin/pip install -r requirements.txt

> ./bin/pip install opencv-python .. since I have not added it to the thing


By this time you should have a good setup for playing with this stuff without messing up your base py setup.


Then to quickly test your setup is working properly:

to test pytorch:
> python3 test-pytorch-setup.py

to test tensorflow:
> python3 test-tensorflow-setup.py


## Thingses
Then we have a couple of ways to denoise:


**Autoencoders** (tensorflow): Should work fine'ish.


**Noise2Noise** (pytorch): A little better - but it's a classic CNN.


**CycleGAN** (tensorflow): The most promising of the three.


**BlindSpot** (pytorch): Another approach using CNN. Very dramatic results.


**cGAN** (pytorch): Conditional GAN. No idea if it works.


**Visualize**: A tool for comparing all the four images.


**Tracker**: A very basic resnet tracker for doing bounding boxes on interesting stuff in the images.


## Caveats:
Note that this is just example code and needs a lot of adaptation for them to work on your datasets.

All of them expect input in data/training and will output in their own directory under data. 


