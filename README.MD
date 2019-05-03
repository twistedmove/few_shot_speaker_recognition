## **Few Shot Speaker Recognition**

### Data
Audio from youtube videos for 5 speakers were used to generate this test dataset. These speakers are Barack Obama, Hillary Clinton, Michelle Obama, Justin Trudeau and Theresa May. These speakers are not present in voxceleb dataset. For each of these speakers, audio from 5 videos were collected and from each audio, three 15 second audio clips were extracted. So, for 5 speakers 75 audio clips are present and list of these clips is present in "data/vid_list.csv". The dataset is downloaded by running the command:
	python download.py

### Model
Model architecture: ResNet-34
Model is pre-trained on first 200 classes of voxceleb dataset for 1-shot and 5-shot speaker recognition using prototypical loss. The dimension of embedding used is 64.

### Usage
First download the pre-trained model for 1-shot and 5-shot recognition available [here](https://drive.google.com/file/d/1pwyq3sxNIIB9HVgTtoLZsHpgIBoqA3Tq/view?usp=sharing) and [here](https://drive.google.com/file/d/10EjtmZJowP64AVKg51iAV4Bn9XIUKpTG/view?usp=sharing) and put both of them in pre_trained folder.

For 1-shot speaker recognition results, run:
	python test.py -s 1

For 5-shot speaker recognition results, run:
	python test.py -s 5

### Results
Accuracy depends on which 1 or 5 samples are chosen randomly to be used as support, hence, accuracy reported is averaged over 10 iteration.
1-shot speaker recognition accuracy: 72%
5-shot speaker recognition accuracy: 88%