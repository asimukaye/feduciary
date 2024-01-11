import torch
import numpy as np
import warnings
import os
from torch import Tensor
import json
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve,\
    average_precision_score, f1_score, precision_score, recall_score,\
        mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,\
            r2_score, d2_pinball_score, top_k_accuracy_score

from .basemetric import BaseMetric

warnings.filterwarnings('ignore')


class Acc1(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self, out_prefix: str = ''):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        self.scores = []
        self.answers = []

        # files = [filename for filename in os.listdir('temp') if filename.startswith(f'{out_prefix}answers')]

        # if files:
        #     files.sort(reverse=True)
        #     last_num = int(files[0].removeprefix(f'{out_prefix}answers_').removesuffix('.json')) + 1
        # else:
        #     last_num = 0

        # with open(f'{out_prefix}answers_{last_num}.json', 'w') as answers_json:
        #     json.dump(answers.tolist(), answers_json)
        # # print(f'scores: {scores.shape}')
        # print(f'answers: {answers.shape}')
        # with open(f'{out_prefix}scores.json', 'w') as scores_json:
        #     json.dump(scores.tolist(), scores_json)
        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
            # files = [filename for filename in os.listdir('temp') if filename.startswith(f'{out_prefix}labels')]

            # if files:
            #     files.sort(reverse=True)
            #     last_num_2 = int(files[0].removeprefix(f'{out_prefix}labels_').removesuffix('.json')) + 1
            # else:
            #     last_num_2 = 0
            # with open(f'{out_prefix}labels_{last_num_2}.json', 'w') as labels_json:
            #     json.dump(labels.tolist(), labels_json)
            # print(f'labels: {labels.shape}')
        else: # binary - use Youden's J to determine label
            scores = scores.sigmoid().numpy()
            fpr, tpr, thresholds = roc_curve(answers, scores)
            cutoff = thresholds[np.argmax(tpr - fpr)]
            labels = np.where(scores >= cutoff, 1, 0)
        return accuracy_score(answers, labels)


class Acc5(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).softmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()

        self.scores = []
        self.answers = []
        num_classes = scores.shape[-1]
        return top_k_accuracy_score(answers, scores, k=5, labels=np.arange(num_classes))

class Auroc(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).softmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []
        num_classes = scores.shape[-1]
        return roc_auc_score(answers, scores, average='weighted', multi_class='ovr', labels=np.arange(num_classes))

class Auprc(BaseMetric): # only for binary classification
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).sigmoid().numpy()
        answers = torch.cat(self.answers).numpy()

        self.scores = []
        self.answers = []
        return average_precision_score(answers, scores, average='weighted')

class Youdenj(BaseMetric):  # only for binary classification
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).sigmoid().numpy()
        answers = torch.cat(self.answers).numpy()

        self.scores = []
        self.answers = []
        fpr, tpr, thresholds = roc_curve(answers, scores)
        return thresholds[np.argmax(tpr - fpr)]

class F1(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        self.scores = []
        self.answers = []
        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: # binary - use Youden's J to determine label
            scores = scores.sigmoid().numpy()
            fpr, tpr, thresholds = roc_curve(answers, scores)
            cutoff = thresholds[np.argmax(tpr - fpr)]
            labels = np.where(scores >= cutoff, 1, 0)
        return f1_score(answers, labels, average='weighted', zero_division=0)

class Precision(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []
        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: # binary - use Youden's J to determine label
            scores = scores.sigmoid().numpy()
            fpr, tpr, thresholds = roc_curve(answers, scores)
            cutoff = thresholds[np.argmax(tpr - fpr)]
            labels = np.where(scores >= cutoff, 1, 0)
        return precision_score(answers, labels, average='weighted', zero_division=0)

class Recall(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []
        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        else: # binary - use Youden's J to determine label
            scores = scores.sigmoid().numpy()
            fpr, tpr, thresholds = roc_curve(answers, scores)
            cutoff = thresholds[np.argmax(tpr - fpr)]
            labels = np.where(scores >= cutoff, 1, 0)
        return recall_score(answers, labels, average='weighted', zero_division=0)

class Seqacc(BaseMetric):
    def __init__(self):
        super().__init__()

    def collect(self, pred: Tensor, true: Tensor):
        num_classes = pred.size(-1)
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p.view(-1, num_classes))
        self.answers.append(t.view(-1))

    def summarize(self):
        labels = torch.cat(self.scores).argmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []
        # ignore special tokens
        labels = labels[answers != -1]
        answers = answers[answers != -1]
        return accuracy_score(answers, labels)

class Mse(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []        
        return mean_squared_error(answers, scores)

class Rmse(Mse):
    def __init__(self):
        super(Rmse, self).__init__()

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []        
        return mean_squared_error(answers, scores, squared=False)

class Mae(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []
        return mean_absolute_error(answers, scores)

class Mape(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []
        return mean_absolute_percentage_error(answers, scores)

class R2(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self, *args):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []
        return r2_score(answers, scores)

class D2(BaseMetric):
    def __init__(self):
        super().__init__()

    def summarize(self, *args):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        self.scores = []
        self.answers = []
        return d2_pinball_score(answers, scores)
