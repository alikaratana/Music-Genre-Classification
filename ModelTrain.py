import numpy
import pandas
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def main():
    data_set = pandas.read_csv('data_set.csv', index_col=False)
    data_set = numpy.array(data_set)
    print("Dataset shape:", data_set.shape)
    number_of_rows, number_of_cols = data_set.shape

    data_x = data_set[:, :number_of_cols - 1]
    data_y = data_set[:, number_of_cols - 1]

    model = SVC(C=100, gamma=0.08)
    print("Training the model.....")
    model.fit(data_x, data_y)

    joblib.dump(model, 'model.pkl')
    print("Trained and saved the model to project folder successfully.")


if __name__ == '__main__':
    main()
