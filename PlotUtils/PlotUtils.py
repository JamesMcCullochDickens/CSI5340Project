import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from matplotlib.ticker import MaxNLocator

def save_confusion_matrix(cf_matrix, save_fp):
    n_classes = cf_matrix.shape[0]
    df_cm = pd.DataFrame(cf_matrix, range(n_classes), range(n_classes))
    sn.heatmap(df_cm)
    plt.savefig(save_fp)
    plt.close()


# assuming y_vals has multiple lists
def multi_plot(x_vals, y_vals, y_labels, title, x_label, y_label, save_fp):
    title = title.replace("_", " ")
    plt.title(title)
    for i in range(len(y_labels)):
        plt.plot(x_vals, y_vals[i], label=y_labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_fp)
    plt.clf()
    plt.close()


def plot(save_fp, x_vals, y_vals, title, x_label, y_label):
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # make sure the x axis uses integers rather than floats
    title = title.replace("_", " ")
    plt.title(title)
    plt.plot(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_fp)
    plt.clf()
    plt.close()