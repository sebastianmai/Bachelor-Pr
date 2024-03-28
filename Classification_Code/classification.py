import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.colors
import numpy as np


matplotlib.use('Qt5Agg')


def lda(data, classes, direction):

    """
    Function that classifies the data using linear discriminant analysis
    """

    X = data.iloc[:, :-1]  # data values
    y = data.iloc[:, -1]   # data classes


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3, stratify=y, shuffle=True)  # create train/test split

    clf = LDA()
    sfs = SequentialFeatureSelector(clf, direction=direction, n_features_to_select=classes)  # use of SFS/SBS
    selected_train = sfs.fit_transform(X_train, y_train)


    feature_names = sfs.get_feature_names_out()

    classify = clf.fit(selected_train, y_train)

    if classes == 2:
        min1, max1 = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        min2, max2 = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

        x1grid = np.arange(min1, max1, 0.01)
        x2grid = np.arange(min2, max2, 0.01)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1, r2))

        yhat = clf.predict(grid)
        zz = yhat.reshape(xx.shape)


    prediction = clf.predict(sfs.transform(X_test))  # prediction by LDA on the data with only the selected features

    cm = confusion_matrix(y_test, prediction)  # calculate the confusion matrix
    acc = accuracy_score(y_test, prediction)  # calculate the accuracy

    print(acc, feature_names)

    acc = round(acc, 4)


    if classes == 2:

        #plot_best_two(classify, X, sfs, feature_names, "LDA", "SFS_Back")

        trans = sfs.transform(X_test)

        plt.contourf(xx, yy, zz, cmap="viridis", alpha=0.7)

        scatter = plt.scatter(trans[:, 0], trans[:, 1], c=y_test, cmap="viridis", edgecolors='k')
        plt.xlim(selected_train[:, 0].min() - 0.2, selected_train[:, 0].max() + 0.2)
        plt.ylim(selected_train[:, 1].min() - 0.2, selected_train[:, 1].max() + 0.2)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        num_classes = 9
        #cbar = plt.colorbar(scatter, ticks=np.arange(0, num_classes, 1), orientation='vertical')
        #cbar.ax.set_yticklabels(['prestimuli', 'stimuli', 'poststimuli'])
        cbar = plt.colorbar(scatter, ticks=np.arange(0, num_classes, 1), orientation='vertical')
        cbar.ax.set_yticklabels(['pre', 'stim_blue', 'post_blue', 'stim_red', 'post_red', 'stim_heat', 'post_heat', 'stim_wind', 'post_wind'])
        #cbar.ax.set_yticklabels(['pre', 'stim_blue', 'post_blue', 'stim_red', 'post_red',])
        plt.title('Decision Boundaries on Test Set')
        #plt.show()

        name = "LDA"
        if direction == "forward":
            plt.savefig(
                f"/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE_RED_HEAT_WIND/PLOTS/P5_P9_SB/{name}/SFS/best_two2.pdf",
                format='pdf')
            plt.close()
        else:
            plt.savefig(
                f"/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE_RED_HEAT_WIND/PLOTS/P5_P9_SB/{name}/SFS_Back/best_two2.pdf",
                format='pdf')
            plt.close()


    # plotting
    if direction == "forward":
        plot_confusion_matrix(cm, feature_names, "LDA", "SFS", classes, acc)
    else:
        plot_confusion_matrix(cm, feature_names, "LDA", "SFS_Back", classes, acc)

def knn(data, classes, direction):

    """
    Function that classifies the data using k-nearest neighbors
    """

    X = data.iloc[:, :-1]   # data values
    y = data.iloc[:, -1]    # data classes



    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3, stratify=y, shuffle=True)  # create train/test split


    clf = KNeighborsClassifier(n_neighbors=3)
    sfs = SequentialFeatureSelector(clf, direction=direction, n_features_to_select=classes)  # use of SFS/SBS
    selected_train = sfs.fit_transform(X_train, y_train)

    feature_names = sfs.get_feature_names_out()
    classify = clf.fit(selected_train, y_train)

    if classes == 2:
        min1, max1 = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        min2, max2 = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

        x1grid = np.arange(min1, max1, 0.01)
        x2grid = np.arange(min2, max2, 0.01)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1, r2))

        yhat = clf.predict(grid)
        zz = yhat.reshape(xx.shape)


    prediction = clf.predict(sfs.transform(X_test))  # prediction by KNN on the data with only the selected features

    cm = confusion_matrix(y_test, prediction)  # calculate the confusion matrix
    acc = accuracy_score(y_test, prediction)  # calculate the accuracy

    print(acc, feature_names)

    acc = round(acc, 4)


    if classes == 2:
        #plot_best_two(classify, X, sfs, feature_names, "KNN", "SFS")

        trans = sfs.transform(X_test)

        plt.contourf(xx, yy, zz, cmap="viridis", alpha=0.7)

        scatter = plt.scatter(trans[:, 0], trans[:, 1], c=y_test, cmap="viridis", edgecolors='k')
        plt.xlim(selected_train[:, 0].min() - 0.2, selected_train[:, 0].max() + 0.2)
        plt.ylim(selected_train[:, 1].min() - 0.2, selected_train[:, 1].max() + 0.2)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        num_classes = 9
        #cbar = plt.colorbar(scatter, ticks=np.arange(0, num_classes, 1), orientation='vertical')
        #cbar.ax.set_yticklabels(['prestimuli', 'stimuli', 'poststimuli'])
        cbar = plt.colorbar(scatter, ticks=np.arange(0, num_classes, 1), orientation='vertical')
        cbar.ax.set_yticklabels(['pre', 'stim_blue', 'post_blue', 'stim_red', 'post_red', 'stim_heat', 'post_heat', 'stim_wind', 'post_wind'])
        #cbar.ax.set_yticklabels(['pre', 'stim_blue', 'post_blue', 'stim_red', 'post_red',])
        plt.title('Decision Boundaries on Test Set')
        #plt.show()

        name = "KNN"
        if direction == "forward":
            plt.savefig(
                f"/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/RED/PLOTS/ALL/{name}/SFS/best_two2.pdf",
                format='pdf')
            plt.close()
        else:
            plt.savefig(
                f"/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/RED/PLOTS/ALL/{name}/SFS_Back/best_two2.pdf",
                format='pdf')
            plt.close()


    if direction == "forward":
        plot_confusion_matrix(cm, feature_names, "KNN", "SFS", classes, acc)
    else:
        plot_confusion_matrix(cm, feature_names, "KNN", "SFS_Back", classes, acc)



def adaclassify(data, classes, direction):

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=3, stratify=y, shuffle=True)

    clf = AdaBoostClassifier(algorithm='SAMME')
    sfs = SequentialFeatureSelector(clf, direction=direction, n_features_to_select=classes)
    selected_train = sfs.fit_transform(X_train, y_train)

    feature_names = sfs.get_feature_names_out()


    clf.fit(selected_train, y_train)

    if classes == 2:
        min1, max1 = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        min2, max2 = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

        x1grid = np.arange(min1, max1, 0.01)
        x2grid = np.arange(min2, max2, 0.01)
        xx, yy = np.meshgrid(x1grid, x2grid)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.hstack((r1, r2))

        yhat = clf.predict(grid)
        zz = yhat.reshape(xx.shape)

    prediction = clf.predict(sfs.transform(X_test))  # prediction by ADA on the data with only the selected features

    cm = confusion_matrix(y_test, prediction)  # calculate the confusion matrix
    acc = accuracy_score(y_test, prediction)  # calculate the accuracy

    print(acc, feature_names)

    acc = round(acc, 4)


    if classes == 2:
        trans = sfs.transform(X_test)

        plt.contourf(xx, yy, zz, cmap="viridis", alpha=0.7)

        scatter = plt.scatter(trans[:, 0], trans[:, 1], c=y_test, cmap="viridis", edgecolors='k')
        plt.xlim(selected_train[:, 0].min() - 0.2, selected_train[:, 0].max() + 0.2)
        plt.ylim(selected_train[:, 1].min() - 0.2, selected_train[:, 1].max() + 0.2)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        num_classes = 9
        #cbar = plt.colorbar(scatter, ticks=np.arange(0, num_classes, 1), orientation='vertical')
        #cbar.ax.set_yticklabels(['prestimuli', 'stimuli', 'poststimuli'])
        cbar = plt.colorbar(scatter, ticks=np.arange(0, num_classes, 1), orientation='vertical')
        cbar.ax.set_yticklabels(['pre', 'stim_blue', 'post_blue', 'stim_red', 'post_red', 'stim_heat', 'post_heat', 'stim_wind', 'post_wind'])
        #cbar.ax.set_yticklabels(['pre', 'stim_blue', 'post_blue', 'stim_red', 'post_red',])
        plt.title('Decision Boundaries on Test Set')
        #plt.show()

        name = "ADA"
        if direction == "forward":
            plt.savefig(
                f"/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE_RED_HEAT_WIND/PLOTS/P5_P9_SB/{name}/SFS/best_two2.pdf",
                format='pdf')
            plt.close()
        else:
            plt.savefig(
                f"/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE_RED_HEAT_WIND/PLOTS/P5_P9_SB/{name}/SFS_Back/best_two2.pdf",
                format='pdf')
            plt.close()



    if direction == "forward":
        plot_confusion_matrix(cm, feature_names, "ADA", "SFS", classes, acc)
    else:
        plot_confusion_matrix(cm, feature_names, "ADA", "SFS_Back", classes, acc)



def custom_cmap():
    return matplotlib.colors.ListedColormap(["lightsalmon", "lightskyblue", "lightgreen"])


def plot_confusion_matrix(cm, feature_names, name, direction, classes, acc):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["prestimuli", "stimuli BLUE", "poststimuli BLUE", "stimuli RED", "poststimuli RED", 'stimuli HEAT', 'poststimuli HEAT', 'stimuli WIND', 'poststimuli WIND'])
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["prestimuli", "stimuli BLUE", "poststimuli BLUE", "stimuli RED", "poststimuli RED"])
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["prestimuli", "stimuli", "poststimuli"])

    disp.plot(cmap='Purples')
    plt.xticks(rotation=45)
    plt.tight_layout()

    s = ""
    for elem in feature_names:
        s = s + "_" + elem

    plt.savefig(f"/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE_RED_HEAT_WIND/PLOTS/P5_P9_SB/{name}/{direction}/{classes}_features_{acc}{s}.pdf", format='pdf')

    #plt.show()
    plt.close()

def plot_best_two(classify, X, sfs, feature_names, name, direction):
    int1, int2, int3= [], [], []
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
    #plt.legend(bbox_to_anchor=(0.5, 0.3), loc='upper center')

    plt.legend(bbox_to_anchor=(0.71, 1.0), loc='upper left')
    plt.savefig(f"/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE/PLOTS/CH1_CH2_SB/{name}/{direction}/best_two2.pdf", format='pdf')
    #plt.show()
    #plt.close()


def combining(CH1, CH2, P5, P9):

    """
    Function that combines that interleaves the columns. Depending on the channel combination either return combined, combined2 or result
    """

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

    result = pd.concat([combined, combined2], ignore_index=True)

    return combined


if __name__ == '__main__':

    # read the features datasets:
    CH1 = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE/CYBRES_BLUE_CH1.csv',
                       usecols=range(1, 11))
    CH2 = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE/CYBRES_BLUE_CH2.csv',
                       usecols=range(1, 11))
    P5 = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE/Phyto_BLUE_P5.csv',
                       usecols=range(1, 11))
    P9 = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/BLUE/Phyto_BLUE_P9.csv',
                       usecols=range(1, 11))

    CH1_RED = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/RED/CYBRES_RED_CH1.csv',
                       usecols=range(1, 11))
    CH2_RED = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/RED/CYBRES_RED_CH2.csv',
                       usecols=range(1, 11))
    P5_RED = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/RED/Phyto_RED_P5.csv',
                       usecols=range(1, 11))
    P9_RED = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/RED/Phyto_RED_P9.csv',
                       usecols=range(1, 11))

    CH1_HEAT = pd.read_csv('/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/HEAT/CYBRES_HEAT_CH1.csv',
                       usecols=range(1, 11))
    CH2_HEAT = pd.read_csv('/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/HEAT/CYBRES_HEAT_CH2.csv',
                       usecols=range(1, 11))
    P5_HEAT = pd.read_csv('/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/HEAT/Phyto_HEAT_P5.csv',
                       usecols=range(1, 11))
    P9_HEAT = pd.read_csv('/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/HEAT/Phyto_HEAT_P9.csv',
                       usecols=range(1, 11))

    CH1_WIND = pd.read_csv('/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/CYBRES_WIND_CH1.csv',
                       usecols=range(1, 11))
    CH2_WIND = pd.read_csv('/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/CYBRES_WIND_CH2.csv',
                       usecols=range(1, 11))
    P5_WIND = pd.read_csv('/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/Phyto_WIND_P5.csv',
                       usecols=range(1, 11))
    P9_WIND = pd.read_csv('/home/bastimai/Data/Universität/Bachelor/Projekt/Bachelor-Pr/Features/WIND/Phyto_WIND_P9.csv',
                       usecols=range(1, 11))



    # calculating the prestimuli dataset
    combine_blue = combining(CH1, CH2, P5, P9)
    combine_red = combining(CH1_RED, CH2_RED, P5_RED, P9_RED)
    combine_red['class'].replace({1: 3, 2: 4}, inplace=True)
    combine_heat = combining(CH1_HEAT, CH2_HEAT, P5_HEAT, P9_HEAT)
    combine_heat['class'].replace({1: 5, 2: 6}, inplace=True)
    combine_wind = combining(CH1_WIND, CH2_WIND, P5_WIND, P9_WIND)
    combine_wind['class'].replace({1: 7, 2: 8}, inplace=True)

    #combine blue, red, heat and wind datasets
    helper = pd.concat([combine_blue, combine_red, combine_heat, combine_wind], ignore_index=True)
    helper.drop(helper.index[::3], inplace=True)


    class_pre = helper.iloc[::3]
    length_classes = min(helper['class'].value_counts())
    class_pre = class_pre.sample(n=length_classes, random_state=42).reset_index(drop=True)
    class_pre.to_csv('class_pre_P5_P9.csv', index=False)


    class_pre_CH1_CH2 = pd.read_csv('class_pre_CH1_CH2.csv')
    class_pre_P5_P9 = pd.read_csv('class_pre_P5_P9.csv')
    #class_pre = class_pre_CH1_CH2
    class_pre = class_pre_P5_P9
    #class_pre = pd.concat([class_pre_CH1_CH2, class_pre_P5_P9], ignore_index=True)


    result = pd.concat([helper, class_pre], ignore_index=True)




    # doing the classification
    dir = "backward"
    for i in range(1, 18):
        num = i
        lda(result, num, dir)
        knn(result, num, dir)
        adaclassify(result, num, dir)

    dir = "forward"
    for i in range(1, 18):
        num = i

        lda(result, num, dir)
        knn(result, num, dir)
        adaclassify(result, num, dir)
