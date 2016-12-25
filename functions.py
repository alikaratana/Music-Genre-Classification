import matplotlib.pyplot as plt, librosa, numpy, sklearn, itertools

def extract_features(signal,sample_rate,frame_size,hop_size):
    
    #Timbral Texture Features
    zero_crossing_rate=librosa.feature.zero_crossing_rate(y=signal,frame_length=frame_size,hop_length=hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal,sr=sample_rate,n_fft=frame_size,hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal,sr=sample_rate,n_fft=frame_size,hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal,sr=sample_rate,n_fft=frame_size,hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal,sr=sample_rate,n_fft=frame_size,hop_length=hop_size)
    mfccs=librosa.feature.mfcc(y=signal,sr=sample_rate,n_fft=frame_size,hop_length=hop_size)
    
    #Rhythm Content Features
   
    #Pitch Content Features
    
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
        
        numpy.mean(mfccs[1,:]),
        numpy.std(mfccs[1,:]),       
        numpy.mean(mfccs[2,:]),
        numpy.std(mfccs[2,:]),        
        numpy.mean(mfccs[3,:]),
        numpy.std(mfccs[3,:]),
        numpy.mean(mfccs[4,:]),
        numpy.std(mfccs[4,:]),
        numpy.mean(mfccs[5,:]),
        numpy.std(mfccs[5,:]),
        numpy.mean(mfccs[6,:]),
        numpy.std(mfccs[6,:]),
        numpy.mean(mfccs[7,:]),
        numpy.std(mfccs[7,:]),
        numpy.mean(mfccs[8,:]),
        numpy.std(mfccs[8,:]),
        numpy.mean(mfccs[9,:]),
        numpy.std(mfccs[9,:]),
        numpy.mean(mfccs[10,:]),
        numpy.std(mfccs[10,:]),
        numpy.mean(mfccs[11,:]),
        numpy.std(mfccs[11,:]),
        numpy.mean(mfccs[12,:]),
        numpy.std(mfccs[12,:]),
        numpy.mean(mfccs[13,:]),
        numpy.std(mfccs[13,:]),
    ]

def create_label_column(number_of_songs,number_of_genres):
    target=[]
    for i in range(0,number_of_songs):
        if(i<(number_of_songs/number_of_genres)):
            target.append('B')
        elif((number_of_songs/number_of_genres)<=i<2*(number_of_songs/number_of_genres)):
            target.append('Cl')
        elif(2*(number_of_songs/number_of_genres)<=i<3*(number_of_songs/number_of_genres)):
            target.append('Co')
        elif(3*(number_of_songs/number_of_genres)<=i<4*(number_of_songs/number_of_genres)):
            target.append('D')
        elif(4*(number_of_songs/number_of_genres)<=i<5*(number_of_songs/number_of_genres)):
            target.append('H')
        elif(5*(number_of_songs/number_of_genres)<=i<6*(number_of_songs/number_of_genres)):
            target.append('J')
        elif(6*(number_of_songs/number_of_genres)<=i<7*(number_of_songs/number_of_genres)):
            target.append('M')
        elif(7*(number_of_songs/number_of_genres)<=i<8*(number_of_songs/number_of_genres)):
            target.append('P')
        elif(8*(number_of_songs/number_of_genres)<=i<9*(number_of_songs/number_of_genres)):
            target.append('Re')
        elif(9*(number_of_songs/number_of_genres)<=i<10*(number_of_songs/number_of_genres)):
            target.append('Ro')
    return target

def confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_cnf(model,dataset_x,dataset_y,GENRES):
    true_y=dataset_y
    true_x=dataset_x
    pred=model.predict(true_x)

    print("---------------PERFORMANCE ANALYSIS FOR THE MODEL----------------\n")

    print("Real Test dataset labels: \n{}\n".format(true_y))
    print("Predicted Test dataset labels: \n{}".format(pred))

    cnf_matrix=sklearn.metrics.confusion_matrix(true_y,pred)
    plt.figure()
    a=confusion_matrix(cnf_matrix,classes=GENRES,title='Confusion matrix')
    


