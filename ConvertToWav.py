import os
from pydub import AudioSegment
import config

INPUT_DIR = config.ConvertWav.CONVERT_DIRECTORY
OUTPUT_DIR = INPUT_DIR + "_wav" + "/"
FILE_TYPE = config.ConvertWav.FILE_TYPE


def main():
    print("Creating new folder...")
    try:
        os.mkdir(OUTPUT_DIR)
        print("Converting files to wav format...")
        for eachfile in os.listdir(INPUT_DIR):
            song = AudioSegment.from_file(INPUT_DIR + "/" + eachfile, FILE_TYPE)
            song.export(OUTPUT_DIR + eachfile[:-3] + "_wav.wav", format="wav")
        print("DONE!")
    except:
        pass


if __name__ == '__main__':
    main()
