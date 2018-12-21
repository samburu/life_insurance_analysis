import itertools
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics
from lifelines import KaplanMeierFitter

def kaplan_curve(df, duration, censorship, cohort=None):
    """ Estimate the survival function
    Parameters:
        df (pd.DataFrame): The dataframe with the necessary columns
        duration (str): The time between birth and death columns
        censorship (str): True if the death was observed else False column
        cohort (pd.Series or None): The different group to be compared (default is None)

    Returns:
        a dict with KaplanMeierFitter
    """
    # List of the neccessary columns
    cols = [col for col in [duration, censorship, cohort] if col is not None]

    # Drop missing values
    df = df.dropna(subset=cols)
    output = {}
    fig, ax = plt.subplots(figsize=[10, 5])
    if not cohort:
        output["kmf"] = KaplanMeierFitter()
        output["kmf"].fit(df[duration], df[censorship])
        output["kmf"].plot(ax=ax)
    else:
        for i, val in enumerate(set(df[cohort])):
            temp = df.loc[df[cohort] ==  val]
            output[f"kmf_{val}"] = KaplanMeierFitter()
            output[f"kmf_{val}"].fit(temp[duration], temp[censorship], label=val)
            output[f"kmf_{val}"].plot(ax=ax)

    return output


def plot_confusion_matrix(y_true, y_pred, lbl_class,
                          normalize=True,
                          savefig=None,
                          title='Confusion matrix for Fundraising prediction',
                          cmap=plt.cm.YlOrRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=lbl_class)
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(lbl_class))
    plt.xticks(tick_marks, lbl_class, rotation=45)
    plt.yticks(tick_marks, lbl_class)

    print(cm)

    thresh = cm.max() / 1.45
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)
