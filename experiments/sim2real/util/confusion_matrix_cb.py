from keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np


class ConfusionMatrixPlotter(Callback):
    """Plot the confusion matrix on a graph and update after each epoch
    # Arguments
        X_val: The input values
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title
    """

    def __init__(self, X_val, Y_val, classes, interactive=True, normalize=False, to_file=None, cmap=plt.cm.Blues,
                 title='Confusion Matrix', remap_classes=None):
        self.X_val = X_val
        self.Y_val = Y_val
        self.title = title
        self.classes = classes
        self.normalize = normalize
        self.cmap = cmap
        self.interactive = interactive
        self.to_file = to_file
        self.remap_classes = remap_classes

        if interactive:
            plt.ion()

        plt.figure(figsize=[6.4 * 2, 4.8 * 2])

        if to_file is None:
            plt.title(self.title)

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch=None, logs={}):
        plt.clf()

        pred = self.model.predict(self.X_val)

        if self.remap_classes is not None:
            pred = self.remap_classes(pred)

        max_pred = np.argmax(pred, axis=1)
        max_y = np.argmax(self.Y_val, axis=1)
        cnf_mat = confusion_matrix(max_y, max_pred)

        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

        thresh = cnf_mat.max() / 2.
        for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
            plt.text(j, i, cnf_mat[i, j],
                     horizontalalignment="center",
                     color="white" if cnf_mat[i, j] > thresh else "black")

        plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)

        # Labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.colorbar()

        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.draw()

        if self.to_file is not None:
            plt.savefig(self.to_file, dpi=150)
        else:
            plt.show()

            if self.interactive:
                plt.pause(0.001)
