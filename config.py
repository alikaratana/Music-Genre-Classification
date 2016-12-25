########## CONFIGURATIONS FOR Convert_to_wav ##########
class ConvertWav:

	## Path of the song folder which will be used for wav converting
	CONVERT_DIRECTORY="./genres/"
	## Name of the folder that includes converted wav files
	OUTPUT_FOLDER_NAME="genres_wav"

########## CONFIGURATIONS FOR Create_Dataset_as_csv ##########
class CreateDataset:

	## Path of GTZAN dataset
	DATASET_DIRECTORY="./genres/"
	## Number of songs
	Number_of_songs=1000
	## Number of genres
	Number_of_genres=10
	## Sampling rate (Hz)
	Sampling_rate=22050
	## Frame size (Samples)
	Frame_size=2048
	## Hop Size (Samples)
	Hop_size=512
	## Window size (Samples)
	Window_size=2048

########## CONFIGURATIONS FOR Work#1 & Work#2 ##########
class Genre2:

	## Array of genre names 
	GENRE_NAMES=["BLUES","CLASSICAL"]
	## Percentage of test size for GTZAN 
	TEST_SIZE=0.30
	## Path for test data
	TEST_DATA_PATH=""

########## CONFIGURATIONS FOR Work#3 ##########
class Genre3:

	## Array of genre names 
	GENRE_NAMES=["METAL","CLASSICAL","HIPHOP"]
	## Percentage of test size for GTZAN 
	TEST_SIZE=0.30
	## Path for test data
	TEST_DATA_PATH="./test3"

########## CONFIGURATIONS FOR Work#4 ##########
class Genre4:

	## Array of genre names 
	GENRE_NAMES=["METAL","CLASSICAL","HIPHOP","BLUES"]
	## Percentage of test size for GTZAN 
	TEST_SIZE=0.30
	## Path for test data
	TEST_DATA_PATH="./test4"

########## CONFIGURATIONS FOR Work#5 ##########
class Genre5:

	## Array of genre names 
	GENRE_NAMES=["METAL","CLASSICAL","HIPHOP","BLUES","POP"]
	## Percentage of test size for GTZAN 
	TEST_SIZE=0.30
	## Path for test data
	TEST_DATA_PATH="./test5"

########## CONFIGURATIONS FOR Work#6 ##########
class Genre6:

	## Array of genre names 
	GENRE_NAMES=["METAL","CLASSICAL","HIPHOP","BLUES","POP","REGGAE"]
	## Percentage of test size for GTZAN 
	TEST_SIZE=0.15
	## Path for test data
	TEST_DATA_PATH="./test6"


########## CONFIGURATIONS FOR Work#7 ##########
class Genre10:

	## Array of genre names 
	GENRE_NAMES=["BLUES","CLASSICAL","COUNTRY","DISCO","HIPHOP","JAZZ","METAL","POP","REGGAE","ROCK"]
	## Percentage of test size for GTZAN 
	TEST_SIZE=0.15
	## Path for test data
	TEST_DATA_PATH="./test10"

