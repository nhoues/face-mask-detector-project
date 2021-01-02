import matplotlib.pyplot as plt 
import numpy as np 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, 'masked', rotation=45)
    plt.yticks(tick_marks, 'masked')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def sigmod(x) : 
    return 1/(1+ np.exp(-x))