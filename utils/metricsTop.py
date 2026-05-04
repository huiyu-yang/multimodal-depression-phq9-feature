import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'CMDC': self.__eval_cmdc_regression
            }
        else:
            self.metrics_dict = {
                'CMDC': self.__eval_cmdc_classification
            }

    def __eval_cmdc_classification(self, y_pred, y_true):
        """
        CMDC binary classification:
          HC=0, MDD=1
        y_pred: (N,2) logits/probs
        y_true: (N,) labels in {0,1}
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy().astype(int).reshape(-1)
        y_pred_2 = np.argmax(y_pred, axis=1)  # predicted class

        acc = accuracy_score(y_true, y_pred_2)
        f1_macro = f1_score(y_true, y_pred_2, average='macro')
        f1_weighted = f1_score(y_true, y_pred_2, average='weighted')
        p_macro, r_macro, _, _ = precision_recall_fscore_support(
            y_true, y_pred_2, average='macro', zero_division=0
        )

        eval_results = {
            "Acc": round(acc, 4),
            "F1_macro": round(f1_macro, 4),
            "F1_weighted": round(f1_weighted, 4),
            "Precision_macro": round(p_macro, 4),
            "Recall_macro": round(r_macro, 4),
        }
        return eval_results


    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_cmdc_regression(self, y_pred, y_true):
        """
        CMDC regression: PHQtotal
        y_pred/y_true: shape (N,1) or (N,)
        """
        pred = y_pred.view(-1).cpu().detach().numpy()
        true = y_true.view(-1).cpu().detach().numpy()

        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        corr = float(np.corrcoef(pred, true)[0][1]) if (np.std(pred) > 1e-8 and np.std(true) > 1e-8) else 0.0

        eval_results = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Corr": round(corr, 4),
        }
        return eval_results
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]