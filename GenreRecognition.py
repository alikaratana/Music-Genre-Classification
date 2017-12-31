import numpy
import librosa
import sklearn
import joblib
import config


def main():
    path = librosa.util.find_files(config.Test.TEST_DATA_PATH)
    sample_rate = config.CreateDataset.SAMPLING_RATE
    hop_size = config.CreateDataset.HOP_SIZE
    frame_size = config.CreateDataset.FRAME_SIZE

    songs = []

    print("Extracting sample arrays for files...")

    for p in path:
        x, sr = librosa.load(p, sr=sample_rate, duration=5.0)
        songs.append(x)

    print("DONE!")

    print("Extracting features from sample arrays...")

    data = numpy.array([extract_features(x, sample_rate, frame_size, hop_size) for x in songs])

    print("DONE!")

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)

    # get the model from pkl file
    svm = joblib.load('model.pkl')

    print("----------------------------------- Predicted Labels -----------------------------------\n")
    preds = svm.predict(data)
    print(preds)
    print("")
    print("----------------------------------------------------------------------------------------")


def extract_features(signal, sample_rate, frame_size, hop_size):

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size,
                                                            hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size,
                                                        hop_length=hop_size)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    return [

        numpy.mean(zero_crossing_rate),
        numpy.std(zero_crossing_rate),
        numpy.mean(spectral_centroid),
        numpy.std(spectral_centroid),
        numpy.mean(spectral_contrast),
        numpy.std(spectral_contrast),
        numpy.mean(spectral_bandwidth),
        numpy.std(spectral_bandwidth),
        numpy.mean(spectral_rolloff),
        numpy.std(spectral_rolloff),

        numpy.mean(mfccs[1, :]),
        numpy.std(mfccs[1, :]),
        numpy.mean(mfccs[2, :]),
        numpy.std(mfccs[2, :]),
        numpy.mean(mfccs[3, :]),
        numpy.std(mfccs[3, :]),
        numpy.mean(mfccs[4, :]),
        numpy.std(mfccs[4, :]),
        numpy.mean(mfccs[5, :]),
        numpy.std(mfccs[5, :]),
        numpy.mean(mfccs[6, :]),
        numpy.std(mfccs[6, :]),
        numpy.mean(mfccs[7, :]),
        numpy.std(mfccs[7, :]),
        numpy.mean(mfccs[8, :]),
        numpy.std(mfccs[8, :]),
        numpy.mean(mfccs[9, :]),
        numpy.std(mfccs[9, :]),
        numpy.mean(mfccs[10, :]),
        numpy.std(mfccs[10, :]),
        numpy.mean(mfccs[11, :]),
        numpy.std(mfccs[11, :]),
        numpy.mean(mfccs[12, :]),
        numpy.std(mfccs[12, :]),
        numpy.mean(mfccs[13, :]),
        numpy.std(mfccs[13, :]),
    ]


if __name__ == '__main__':
    main()
