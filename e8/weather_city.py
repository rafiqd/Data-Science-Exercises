import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def main(labled, unlabled, output):
    data = pd.read_csv(labled)
    scale = StandardScaler()
    X = scale.fit_transform(data[data.columns.values[1:]].values)
    y = data['city'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = make_pipeline(
        SVC(kernel='linear', C=10)
    )
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    unlabled_data = pd.read_csv(unlabled)
    X2 = unlabled_data[unlabled_data.columns.values[1:]].values
    predictions = model.predict(scale.fit_transform(X2))
    pd.Series(predictions).to_csv(output, index=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])