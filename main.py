from __future__ import division
import authors
import methods

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

# Define Important Library
#====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Start from 1 always, no random state
#====================================================================
#np.random.seed(seed=123)

# scikit-learn library import
#====================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import interp

# Necessary Library for Calculation
#====================================================================
from sklearn.metrics import accuracy_score, confusion_matrix, \
    roc_auc_score,roc_curve,auc, average_precision_score, f1_score
#====================================================================
input_file_name = "Data/BenchmarkData70.txt"
features_file_name = "Data/Features.csv"

# Here we can generate our features set using different length
#====================================================================
if(1):
    from feature_27_extractor import generator
    #from feature_extractor_window import generator
    F1=1; F2=1; F3=1; F4=1; F5=1; F6=1; F7=1; F8=1; F9=1;
    F10=1; F11=1; F12=1; F13=1; F14=1; F15=1; F16=1; F17=1; F18=1;
    F19=1; F20=1; F21=1; F22=1; F23=1; F24=1; F25=1; F26=1; F27=1;
    feature_list = [F1, F2, F3, F4, F5, F6, F7, F8, F9,
                    F10, F11, F12, F13, F14, F15, F16, F17, F18,
                    F19, F20, F21, F22, F23, F24, F25, F26, F27]
    generator(input_file_name, features_file_name, feature_list)

# Load the Featureset:
#====================================================================
D = pd.read_csv(features_file_name, header=None)
#D = D.drop_duplicates()  # Return : each row are unique value

# Divide features (X) and classes (y) :
#====================================================================
X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

print ('-> Total Dataset : {}'.format(len(X)))
print ('-> Total features: {}'.format(len(X[1])))
print ('-> Start classification   ...')

# Encoding y :
#====================================================================
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)


# Define classifiers within a list
#====================================================================
Classifiers = [
    SVC(kernel='rbf', C=4, probability=True, decision_function_shape='ovo', tol=0.1, cache_size=200),
    LogisticRegression(n_jobs=1000),
    KNeighborsClassifier(n_jobs=500),
    DecisionTreeClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(n_components=500),
]
ClassifiersName = [
    'SVM','LR','KNC','DTC','GNB','LDA'
]

# Spliting with 10-FCV :
#====================================================================
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)

# Plot margin line for ROC
#====================================================================
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)


# Box plot
#====================================================================
from ploter import plot_BoxPlot, plot_auRoc
acc_boxplot = []
colors = ['pink', 'lightblue', 'lightgreen','cyan', 'tan', 'blue']

# Pick all classifier within the Classifier list and test one by one
#====================================================================
for classifier, cls_name in zip(Classifiers,ClassifiersName):
    # CM = Confusion Matrix
    CM = np.zeros((2, 2), dtype=int)
    accuracy = []
    auroc = []
    aupr = []
    F1 = []
    fold = 1
    tprs = []
    aucs = []
    acc = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    print('____________________________________________')
    print('Classifier: '+classifier.__class__.__name__)
    model = classifier
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # Scaling the feature
        # -----------------------------------------------------------------
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scale = StandardScaler()
        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)
        #print X_test
        # After getting best feature's indices
        # -----------------------------------------------------------------
        #from feature_selector import best_index
        #indices = best_index()
        #X_train = X_train[:, indices]
        #X_test = X_test[:, indices]


        # print the fold number and numbber of feature after selection
        # -----------------------------------------------------------------
        print ('F{} '.format(fold), end="")

        # Train model
        # -----------------------------------------------------------------
        model.fit(X_train, y_train)

        # Evalution
        # -----------------------------------------------------------------
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]


        # Computer ROC
        #---------------------------------------------------------------------
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (fold, roc_auc))

        accuracy.append(accuracy_score(y_pred=y_pred, y_true=y_test))
        auroc.append(roc_auc_score(y_true=y_test, y_score=y_proba))
        aupr.append(average_precision_score(y_true=y_test, y_score=y_proba))
        F1.append(f1_score(y_pred=y_pred, y_true=y_test))
        CM += confusion_matrix(y_pred=y_pred, y_true=y_test)
        fold += 1

    acc_boxplot.append(accuracy)
    print ('')

    TN, FP, FN, TP = CM.ravel()

    print('--------------------------------------------')
    print('| Acc |auROC|auPR | Sp  | Sn  | MCC | F1  |')
    print('--------------------------------------------')
    print('|{:.2f}'.format((np.mean(accuracy) * 100)), end="")
    print('|{:.3f}'.format((np.mean(auroc))), end="")
    print('|{:.3f}'.format((np.mean(aupr))), end="")
    print('|{:.2f}'.format(((TN / (TN + FP)) * 100)), end="")
    print('|{:.2f}'.format(((TP / (TP + FN)) * 100)), end="")
    print('|{:.3f}'.format(((TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))))), end="")
    print('|{:.3f}|'.format(np.mean(F1)))
    print('--------------------------------------------')
    print('Confusion Matrix:{}', CM)
    print('TN: {} FP: {} FN: {} TP: {}'.format(TN, FP, FN, TP))
    CM = np.zeros((2, 2), dtype=int)

    # Call function for ROC plot
    from ploter import plot_auRoc
    plot_auRoc(tprs, mean_fpr,cls_name)


plt.savefig('main_wfs_roc1.png')

for i in range(len(acc_boxplot)):
    for j in range(len(acc_boxplot[i])):
        acc_boxplot[i][j] *= 100.0

#plot_BoxPlot("main_window_box.png", acc_boxplot, ClassifiersName, colors)
plot_BoxPlot("main_wfs_box1.png", acc_boxplot, ClassifiersName, colors)
plt.show()
