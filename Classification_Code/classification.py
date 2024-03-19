import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.colors
import numpy as np

#matplotlib.use('Qt5Agg')


def lda(data, classes, direction):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3, stratify=y, shuffle=True)


    clf = LDA()
    sfs = SequentialFeatureSelector(clf, direction=direction, n_features_to_select=classes)
    selected_train = sfs.fit_transform(X_train, y_train)

    feature_names = sfs.get_feature_names_out()

    classify = clf.fit(selected_train, y_train)
    prediction = clf.predict(sfs.transform(X_test))

    cm = confusion_matrix(y_test, prediction)
    acc = accuracy_score(y_test, prediction)

    print(acc, feature_names)

    acc = round(acc, 4)

    if direction == "forward":
        plot_confusion_matrix(cm, feature_names, "LDA", "SFS", classes, acc)
    else:
        plot_confusion_matrix(cm, feature_names, "LDA", "SFS_Back", classes, acc)

    if classes == 2:
        if direction == "forward":
            plot_best_two(classify, X, sfs, feature_names, "LDA", "SFS")
        else:
            plot_best_two(classify, X, sfs, feature_names, "LDA", "SFS_Back")


def knn(data, classes, direction):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3, stratify=y, shuffle=True)

    clf = KNeighborsClassifier(n_neighbors=3)
    sfs = SequentialFeatureSelector(clf, direction=direction, n_features_to_select=classes)
    selected_train = sfs.fit_transform(X_train, y_train)

    feature_names = sfs.get_feature_names_out()


    classify = clf.fit(selected_train, y_train)
    prediction = clf.predict(sfs.transform(X_test))

    cm = confusion_matrix(y_test, prediction)
    acc = accuracy_score(y_test, prediction)

    print(acc, feature_names)

    acc = round(acc, 4)

    if direction == "forward":
        plot_confusion_matrix(cm, feature_names, "KNN", "SFS", classes, acc)
    else:
        plot_confusion_matrix(cm, feature_names, "KNN", "SFS_Back", classes, acc)

    if classes == 2:
        if direction == "forward":
            plot_best_two(classify, X, sfs, feature_names, "KNN", "SFS")
        else:
            plot_best_two(classify, X, sfs, feature_names, "KNN", "SFS_Back")

def adaclassify(data, classes, direction):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3, stratify=y, shuffle=True)
    print(len(X_test))

    clf = AdaBoostClassifier(algorithm='SAMME')
    sfs = SequentialFeatureSelector(clf, direction=direction, n_features_to_select=classes)
    selected_train = sfs.fit_transform(X_train, y_train)

    feature_names = sfs.get_feature_names_out()


    classify = clf.fit(selected_train, y_train)
    prediction = clf.predict(sfs.transform(X_test))


    cm = confusion_matrix(y_test, prediction)
    acc = accuracy_score(y_test, prediction)

    print(acc, feature_names)

    acc = round(acc, 4)

    if direction == "forward":
        plot_confusion_matrix(cm, feature_names, "ADA", "SFS", classes, acc)
    else:
        plot_confusion_matrix(cm, feature_names, "ADA", "SFS_Back", classes, acc)

    if classes == 2:
        if direction == "forward":
            plot_best_two(classify, X, sfs, feature_names, "ADA", "SFS")
        else:
            plot_best_two(classify, X, sfs, feature_names, "ADA", "SFS_Back")
def custom_cmap():
    return matplotlib.colors.ListedColormap(["lightsalmon", "lightskyblue", "lightgreen"])

def plot_confusion_matrix(cm, feature_names, name, direction, classes, acc):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["prestimuli", "stimuli", "poststimuli"])
    disp.plot(cmap='Purples')
    plt.tight_layout()

    s = ""
    for elem in feature_names:
        s = s + "_" + elem

    plt.savefig(f"/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/PLOTS/ALL_SB/{name}/{direction}/{classes}_features_{acc}{s}.pdf", format='pdf')

    #plt.show()
    plt.close()

def plot_best_two(classify, X, sfs, feature_names, name, direction):
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
    plt.legend(bbox_to_anchor=(0.71, 1.0), loc='upper left')
    plt.savefig(f"/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/PLOTS/ALL_SB/{name}/{direction}/best_two.pdf", format='pdf')
    #plt.show()
    plt.close()



if __name__ == '__main__':
    CH1 = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/CYBRES_WIND_CH1.csv',
                       usecols=range(1, 11))
    CH2 = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/CYBRES_WIND_CH2.csv',
                       usecols=range(1, 11))
    P5 = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/Phyto_WIND_P5.csv',
                       usecols=range(1, 11))
    P9 = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/Phyto_WIND_P9.csv',
                       usecols=range(1, 11))

    print(CH1.keys())
    '''
    # interleaf the columns
    for key in CH1.columns:
        CH1 = CH1.rename(columns={key: f"{key}_s"})
        CH2 = CH2.rename(columns={key: f"{key}_b"})
        P9 = P9.rename(columns={key: f"{key}_s"})
        P5 = P5.rename(columns={key: f"{key}_b"})

    combined = pd.concat((CH1, CH2), axis=1)
    order = np.array(list(zip(CH1.columns, CH2.columns))).flatten()
    combined = combined[order].drop(columns=['class_b']).rename(columns={'class_s': 'class'})

    combined2 = pd.concat((P9, P5), axis=1)
    order = np.array(list(zip(P9.columns, P5.columns))).flatten()

    combined2 = combined2[order].drop(columns=['class_b']).rename(columns={'class_s': 'class'})

    #result = combined
    result = pd.concat([combined, combined2], ignore_index=True)'''

    class_pre = CH1.iloc[::3]
    CH1.drop(CH1.index[::3], inplace=True)

    CH1.reset_index(drop=True, inplace=True)
    length_classes = len(CH1.iloc[::2])

    class_pre = class_pre.sample(n=length_classes).reset_index(drop=True)
    result = class_pre
    #result = pd.concat([CH1, CH2], ignore_index=True)
    print(result)

    '''
    dir = "backward"
    for i in range(1, 9):
        num = i

        lda(result, num, dir)
        knn(result, num, dir)
        adaclassify(result, num, dir)

    dir = "forward"
    for i in range(1, 9):
        num = i

        lda(result, num, dir)
        knn(result, num, dir)
        adaclassify(result, num, dir)'''
