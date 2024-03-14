import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.colors

matplotlib.use('Qt5Agg')


def lda(data, classes):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3, stratify=y, shuffle=True)

    clf = LDA()
    sfs = SequentialFeatureSelector(clf, direction="forward", n_features_to_select=classes)
    #sfs = RFE(clf, n_features_to_select=classes)
    #sfs = RFECV(clf,  scoring='accuracy')
    selected_train = sfs.fit_transform(X_train, y_train)

    feature_names = sfs.get_feature_names_out()

    classify = clf.fit(selected_train, y_train)
    prediction = clf.predict(sfs.transform(X_test))

    cm = confusion_matrix(y_test, prediction)
    acc = accuracy_score(y_test, prediction)

    print(acc, feature_names)

    plot_confusion_matrix(cm)

    if classes == 2:
        plot_best_two(classify, X, sfs, feature_names)


def knn(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3, stratify=y, shuffle=True)

    clf = KNeighborsClassifier(n_neighbors=3)
    sfs = SequentialFeatureSelector(clf, direction="forward", n_features_to_select=2)
    selected_train = sfs.fit_transform(X_train, y_train)

    feature_names = sfs.get_feature_names_out()


    classify = clf.fit(selected_train, y_train)
    prediction = clf.predict(sfs.transform(X_test))

    cm = confusion_matrix(y_test, prediction)
    acc = accuracy_score(y_test, prediction)

    #print(acc)

    plot_confusion_matrix(cm)


def custom_cmap():
    return matplotlib.colors.ListedColormap(["lightsalmon", "lightskyblue", "lightgreen"])

def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["prestimuli", "stimuli", "poststimuli"])
    disp.plot(cmap='Purples')
    plt.tight_layout()
    plt.show()

def plot_best_two(classify, X, sfs, feature_names):
    int1, int2, int3 = [], [], []
    for i in range(0, len(X), 3):
        int1.append(((X[feature_names[0]][i]), (X[feature_names[1]][i])))
        int2.append(((X[feature_names[0]][i+1]), (X[feature_names[1]][i+1])))
        int3.append(((X[feature_names[0]][i+2]), (X[feature_names[1]][i+2])))

    eps = 0.1

    disp = DecisionBoundaryDisplay.from_estimator(classify, sfs.transform(X), grid_resolution=2000, response_method="predict", cmap=custom_cmap())
    disp.ax_.set_xlim([X[feature_names[0]].min() - eps, X[feature_names[0]].max() + eps])
    disp.ax_.set_ylim([X[feature_names[1]].min() - eps, X[feature_names[1]].max() + eps])
    for i in range(len(int1)):
        disp.ax_.scatter(int1[i][0], int1[i][1], marker='o', color='red', label='prestimuli' if i == 0 else "", s=60)
        disp.ax_.scatter(int2[i][0], int2[i][1], marker='x', color='blue', label='stimuli' if i == 0 else "", s=60)
        disp.ax_.scatter(int3[i][0], int3[i][1], marker='1', color='green', label='poststimuli' if i == 0 else "", s=100)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(bbox_to_anchor=(0.7, 0.8), loc='upper left')
    plt.show()

if __name__ == '__main__':
    d = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE/CYBRES_BLUE_CH1.csv',
                       usecols=range(1, 11))
    lda(d, 1)
    #knn(data)

