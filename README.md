### MUSIC GENRE CLASSIFICATION WITH MACHINE LEARNING TECHNIQUES	

#### To use this work on your researches or projects you need:
* Python 3.5.2
* Python packages:
	* IPython
	* Numpy
	* Scipy
	* Pandas
	* Scikit-learn
	* Librosa
	* Matplotlib
* Jupyter Notebook (with IPython kernel)
	
**To install Python:**

_First, check if you already have it installed or not_.
~~~~
python3 --version
~~~~
_If you don't have python 3 in your computer you can use the code below_:
~~~~
sudo apt-get update
sudo apt-get install python3
~~~~

**To install packages via pip install:**
~~~~
sudo pip3 install ipython scipy numpy pandas scikit-learn librosa matplotlib jupyter
~~~~
_If you haven't installed pip, you can use the codes below in your terminal_:
~~~~
sudo apt-get update
sudo apt install python3-pip
~~~~
_You should check and update your pip_:
~~~~
pip3 install --upgrade pip
~~~~

### INFORMATION ABOUT THE REPOSITORY 
**The repository includes:**
* A python file that provides converting audio files to .wav audio format.
* A python file for configurations such as the path of training and test datasets, genre names, number of songs and etc.
* A python file which includes all the necessary functions for classification process.
* A jupyter notebook file that do signal processing and feature extraction from songs and generates a .csv file.
* Several jupyter notebooks for classification process with different numbers of genres.
* A bonus notebook which includes useful information about signal processing !!

**To use the repository:**
* First, run create_dataset_as_csv file and generate a text based dataset as a csv file.
* Then run one of the notebooks for classification based on your choice.
* If you are working with compressed type audio files such as mp3, you may think to convert your files to wav audio format for
 better results which you can do it by using Convert_to_wav.ipynb file.

**Wanna play with data or files ? Feel free !**
* You can change the properties from config.py file as the way you wanted.
* You can add more features or delete some features from being writed to dataset. The function _"extract_features"_ in functions.py
 extract features from songs and you can change the code with your own features.
* You can use your test files

### Note !!

* For now, use can only use the GTZAN dataset for training process.
* Also, you can only use .au files while converting files into .wav.




