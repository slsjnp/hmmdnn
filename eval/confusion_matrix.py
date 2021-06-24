import torch
import numpy as np


class base_confusion:
    def __init__(self, n_classes):
        self.confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype='int')

    def upload_confusion_matrix(self, y_true, y_pred, mat):
        y_pred = (y_pred > 0.5).view(-1).int().cpu()
        y_true = y_true.view(-1).int().cpu()
        matrix = mat
        stacked = np.stack(
            (
                y_true,
                y_pred
            )
            , axis=1
        )
        for i in stacked:
            t, p = i.tolist()
            matrix[t, p] = matrix[t, p] + 1
        self.confusion_matrix = matrix
        return self.confusion_matrix

    # def

# def perdict_measure(y_true, y_pred):
#     TP, FP, TN, FN = 0, 0, 0, 0
#     for i in range(len(y_true)):
#         if y_true(i) == 1 and y_pred == 1:
#             TP += 1
#         if y_true(i) == 1 and y_pred == 0:
#             FN += 1
#         if y_true(i) == 0 and y_pred == 1:
#             FP += 1
#         if y_true(i) == 0 and y_pred == 0:
#             TN += 1
