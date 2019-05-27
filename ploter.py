from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
def plot_auRoc(tprs, mean_fpr, cls_name):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    roc_auc = auc(mean_fpr, mean_tpr)
    #roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(mean_fpr, mean_tpr, label=cls_name+' = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    #plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()



def plot_auRocMean(tprs, mean_fpr, aucs, classifier):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    #plt.plot(mean_fpr, mean_tpr, color='darkorange',
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(classifier+'\nReceiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('main_window_roc2.png')
    plt.show()

# https://matplotlib.org/gallery/statistics/boxplot_color.html#sphx-glr-gallery-statistics-boxplot-color-py
import seaborn as sns
def plot_BoxPlot(title, acc_results, classifier_name, colors):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    labels = classifier_name
    green_diamond = dict(markerfacecolor='r')
    # We can plot multiple box plot such as bplot1, bplot2 ..
    #axes = sns.stripplot(data=acc_results, color="orange", jitter=0.2, size=2.5)
    bplot = axes.boxplot(acc_results,
                         vert=True,
                         patch_artist=True,
                         labels=labels,
                         #notch=True,
                         flierprops=green_diamond,
                        )
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)


    axes.set_title('Box Plot Results')
    # for ax in axes:
    axes.yaxis.grid(True)
    axes.set_xlabel('Classifiers')
    axes.set_ylabel('Accuracy (%)')
    plt.savefig(title, dpi=100)
    #plt.show()

# From Rafsan
def boxPlot(Results, Names):
    ### Algoritms Comparison ###
    # boxplot algorithm comparison
    fig = plt.figure()
    # fig.suptitle('Classifier Comparison')
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)
    plt.boxplot(Results, patch_artist=True, vert = True, whis=True, showbox=True)
    ax.set_xticklabels(Names)
    plt.xlabel('\nName of Classifiers')
    plt.ylabel('\nAccuracy (%)')

    plt.savefig('AccuracyBoxPlot.png', dpi=100)
    plt.show()
    ### --- ###