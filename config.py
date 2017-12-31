

# --- CONFIGURATIONS FOR Convert_to_wav --- #
class ConvertWav:
    # Path of the song folder which will be used for wav converting
    CONVERT_DIRECTORY = "./TestFiles"

    # File extension in song folder
    FILE_TYPE = "au"


# --- CONFIGURATIONS FOR Create_Dataset_as_csv --- #
class CreateDataset:
    # Path of GTZAN dataset
    DATASET_DIRECTORY = "./TrainFiles/"

    # Sampling rate (Hz)
    SAMPLING_RATE = 22050

    # Frame size (Samples)
    FRAME_SIZE = 2048

    # Hop Size (Samples)
    HOP_SIZE = 512


class Test:
    # Path for test data
    TEST_DATA_PATH = "./TestFiles/"

