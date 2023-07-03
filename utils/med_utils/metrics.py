# Copyright (C) 2023 Langlois Quentin, ICTEAM, UCLouvain. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf

from sklearn.metrics import multilabel_confusion_matrix

from utils.med_utils.label_utils import TOTALSEGMENTATOR_LABELS, transform_mask

def compute_confusion_matrix(y_true, y_pred, labels):
    """
        Computes the multi label confusion matrix (with `sklearn.metrics.multilabel_confusion_matrix`), and formats the result
        
        Arguments :
            - y_true : `tf.Tensor / np.ndarray`, the ground truth labels (4-D is interpreted as one-hot encoded)
            - y_pred : `tf.Tensor / np.ndarray`, the predicted labels (4-D is interpreted as one-hot encoded)
            - labels : list of labels, used as keys in the output `dict`
        Return :
            - cm_per_label : dict where keys are labels (in `labels`) and values is a `dict` containing 4 keys : `tp, fp, tn, fn`
        
        **warning** some labels (the last ones) may not be in the result if they are not predicted / expected
    """
    y_true = transform_mask(y_true, 'dense', is_one_hot = len(y_true.shape) == 4)
    y_pred = transform_mask(y_pred, 'dense', is_one_hot = len(y_pred.shape) == 4)
    
    if hasattr(y_true, 'numpy'): y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'): y_pred = y_pred.numpy()

    cm = multilabel_confusion_matrix(y_true.reshape([-1]), y_pred.reshape([-1]))
    return {
        label : {
            'tp' : cm[i, 1, 1], 'fp' : cm[i, 0, 1], 'fn' : cm[i, 1, 0], 'tn' : cm[i, 0, 0] 
        } for i, label in enumerate(labels) if i < len(cm)
    }


def compute_metrics(metrics, metric_name, labels = TOTALSEGMENTATOR_LABELS, ids = None):
    """
        Computes a given metric (e.g. `dice`) based on the multilabel confusion matrix
        
        Arguments :
            - metrics : `dict` containing the multilabel confusion matrix for each `id` in the dataset
                        The structure should be : {
                            id_0 : {
                                label_0 : {tp:, tn:, fp:, fn:},
                                label_1 : {tp:, tn:, fp:, fn:},
                                ...
                            },
                            ...
                        }
                        The nested dict `{label : {tp, fp, fn, tn}}` is the output of `compute_confusion_matrix(...)`
            - metric_name : a `callable` or a valid metric (to add a new metric, add its name / function mapping in the `_metrics_methods`)
            - labels      : the labels to consider (the others are skipped)
            - ids         : data ids to consider (the others are skipped)
        Return :
            - metrics : `dict` where keys are the labels and the value is a list of metric value (1 per id where the label is there)
    """
    if not callable(metric_name) and metric_name not in _metrics_methods:
        raise ValueError('Unsupported metric !\n  Accepted : {}\n  Got : {}'.format(tuple(_metrics_methods.keys()), metric_name))
    
    metric_fn = _metrics_methods[metric_name.lower()] if not callable(metric_name) else metric_name
    
    results = {m if m else 'background' : [] for m in labels}
    for subj_id, infos in metrics.items():
        if ids and subj_id not in ids: continue
        
        for label, cm in infos.items():
            if label in (None, 'null'): label = 'background'
            if label not in results: continue
            
            results[label].append(metric_fn(** cm))
    
    return {c : [vi for vi in v if vi is not None] for c, v in results.items() if v}

def dice_coeff(tp, fp, fn, tn):
    if tp + fn + fp == 0: return None
    inter = tp
    union = 2 * tp + fp + fn
    return 2. * inter / max(1, union)

_metrics_methods = {
    'dice' : dice_coeff
}
