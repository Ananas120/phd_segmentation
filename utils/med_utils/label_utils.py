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

TOTALSEGMENTATOR_LABELS = [
    None, 'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver', 'stomach', 'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left', 'lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1', 'esophagus', 'trachea', 'heart_myocardium', 'heart_atrium_left', 'heart_ventricle_left', 'heart_atrium_right', 'heart_ventricle_right', 'pulmonary_artery', 'brain', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right', 'small_bowel', 'duodenum', 'colon', 'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12', 'humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'clavicula_left', 'clavicula_right', 'femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum', 'face', 'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right', 'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right', 'urinary_bladder'
]

def add_zero_label(mask, is_one_hot):
    if isinstance(mask, tf.sparse.SparseTensor):
        new_indices, new_values, new_shape = mask.indices, mask.values, mask.dense_shape
        if is_one_hot:
            new_shape   = tuple(new_shape[:-1]) + (new_shape[-1] + 1, )
            new_indices = new_indices + tf.cast([
                [0] * (new_indices.shape[1] - 1) + [1]
            ], mask.indices.dtype)
        else:
            new_values = new_values + 1
        
        return tf.sparse.SparseTensor(
            indices = new_indices, values = new_values, dense_shape = new_shape
        )
    elif is_one_hot:
        pad_fn = tf.pad if isinstance(mask, tf.Tensor) else np.pad
        
        padding = [(0, 0)] * (len(mask.shape) - 1) + [(1, 0)]
        return pad_fn(mask, padding)
    else:
        raise RuntimeError('To add a zero-label, the array must either be a Sparse tensor, either one-hot encoded')

def build_mapping(labels, groups):
    """
        Creates a mapping (dict) `{sub_label : index}` where :
        - `sub_label` is a label in `groups` and
        - `index` is its corresponding index from `labels`
        
        If `labels` is not provided, `groups` may be a list of list, where each label in a nested list
        is mapped to its index in the main list
        Example [('a', 'b'), 'c'] -> {'a' : 0, 'b' : 0, 'c' : 1}
        
        Arguments :
            - labels : a list of labels (typically the labels used by the model)
            - groups : a mapping that matches one "main label" to multiple "sub labels"
                - dict : {main_label : list_of_sub_labels}
                - list : nested list, where the main list must have the same length as `labels` (if provided)
                    Example : `labels = ['a', 'b'], groups = [('a1', 'a2'), 'b']`
        Return :
            - dict : where keys are sub-labels, and values are the corresponding "main label" index
    """
    if labels is None and groups is None: return None

    mapping = {}
    if labels is None:
        if isinstance(groups, dict):
            if all(isinstance(v, int) for v in groups.values()): return groups
            groups = list(groups.values())
        
        assert isinstance(groups, list), 'When `labels is None`, you must proved `groups` as a nested list'
        for i, l in enumerate(groups):
            if not isinstance(l, (list, tuple)): l = [l]
            for l_i in l: mapping[l_i] = i
    elif groups is not None:
        if isinstance(groups, list):
            assert len(groups) == len(labels), 'When `groups` is a nested list, it must have the same length as `labels` ({} vs {})'.format(len(groups), len(labels))
            groups = {label : group for label, group in zip(labels, groups)}
        
        for main_label, sub_labels in groups.items():
            if main_label not in labels: continue
            if not isinstance(sub_labels, (list, tuple)): sub_labels = [sub_labels]
            for sub_l in sub_labels: mapping[sub_l] = labels.index(main_label)
    else:
        mapping = {label : i for i, label in enumerate(labels)}
    
    return mapping

def transform_mask(mask, mode, is_one_hot, max_depth = -1):
    if max_depth == -1 and is_one_hot: max_depth = mask.shape[-1]
    is_sparse   = isinstance(mask, tf.sparse.SparseTensor)
    
    if 'dense' in mode and is_one_hot:
        if is_sparse:
            with tf.device('cpu'):
                mask = tf.argmax(tf.sparse.to_dense(mask), axis = -1, output_type = tf.int32)
        elif isinstance(mask, tf.Tensor):
            mask = tf.argmax(mask, axis = -1, output_type = tf.int32)
        else:
            mask = tf.cast(np.argmax(mask, axis = -1), tf.int32)
        
        is_one_hot, is_sparse = False, False

    if 'sparse' in mode and not is_sparse:
        is_sparse   = True
        mask        = tf.sparse.from_dense(mask)
    
    if 'one_hot' in mode and not is_one_hot:
        is_one_hot  = True
        if is_sparse:
            if max_depth == -1: max_depth = tf.reduce_max(mask.values) + 1
            mask = tf.sparse.SparseTensor(
                indices = tf.concat([
                    mask.indices, tf.expand_dims(tf.cast(mask.values, mask.indices.dtype), axis = -1)
                ], axis = -1),
                values = tf.ones((len(mask.indices), ), dtype = tf.uint8),
                dense_shape = tuple(mask.dense_shape) + (max_depth, )
            )
        elif isinstance(mask, tf.Tensor):
            if max_depth == -1: max_depth = tf.reduce_max(mask) + 1
            mask    = tf.cast(tf.math.equal(
                tf.expand_dims(mask, axis = -1), tf.reshape(
                    tf.arange(max_depth, dtype = mask.dtype), [1] * len(mask.shape) + [max_depth]
                )
            ), tf.uint8)
        else:
            if max_depth == -1: max_depth = np.max(mask) + 1
            mask    = tf.cast(np.equal(
                np.expand_dims(mask, axis = -1),
                np.arange(max_depth).reshape([1] * len(mask.shape) + [max_depth])
            ), tf.uint8)

    return mask

def rearrange_labels(array, labels, mapping, default = 0, is_one_hot = None, ** kwargs):
    """
        Rearrange the labels given an initial order (`labels`) and a mapping
        
        Arguments :
            - array : the labels to rearrange (supports np.ndarray and tf.sparse.SparseTensor)
                if `array.shape[-1] == len(labels)`, the array is considered as one-hot encoded
            - labels    : the list of original labels (i.e. the associated label to any value in `array`)
            - mapping   : a mapping to map each label in `labels` to a new id
                - list (of str or list) : each sub-label (i.e. those in the nested lists) are mapped to their list index
                - dict : each key is a label, and the value is its corresponding index
            - default   : the default label to set if no mapping is found
            - kwargs    : additional possible kwargs (mainly ignored)
        Return :
            - the rearranged labels
        
        Important note :
            if `array` is not a `tf.sparse.SparseTensor`:
                The returned value is not a one-hot encoded version (even if the input was)
            else:
                All values mapped to `default` are removed from the `Sparse` array
        
        Example : 
            array = np.array([
                [1, 2, 1],
                [0, 0, 2]
                [3, 1, 2]
            ])
            labels  = ['l0', 'l1', 'l2', 'l3']
            # maps 'l3' from index 3 to 1, and maps 'l1' and 'l2' (index 1 and 2) to index 2
            mapping = ['l0', 'l3', ('l1', 'l2')]

            result == np.array([
                [2, 2, 2],
                [0, 0, 2],
                [1, 2, 2]
            ])
    """
    if is_one_hot is None: is_one_hot  = array.shape[-1] == len(labels)
    is_sparse   = isinstance(array, tf.sparse.SparseTensor)

    list_mapping = mapping if isinstance(mapping, list) else sorted(mapping.keys(), key = mapping.get)
    if list_mapping == labels:
        return array
    elif list_mapping[0] is None and list_mapping[1 : len(labels) + 1] == labels:
        return add_zero_label(array, is_one_hot)
    
    if is_sparse:
        if is_one_hot:
            fn = rearrange_labels_sparse_one_hot
        else:
            fn = rearrange_labels_sparse
    else:
        if is_one_hot:
            fn = rearrange_labels_one_hot
        else:
            fn = rearrange_labels_dense
    
    return fn(array, labels, mapping, default = default, ** kwargs)

def rearrange_labels_one_hot(array, labels, mapping, default = 0, ** kwargs):
    if isinstance(array, tf.Tensor): array = array.numpy()
    array = array.astype(bool)

    val_to_idx = mapping
    if isinstance(mapping, list):
        val_to_idx = {}
        for i, l in enumerate(mapping):
            if not isinstance(l, (list, tuple)): l = [l]
            for l_i in l: val_to_idx[l_i] = i

    result = np.full(array.shape[:-1], default)
    for idx, label in enumerate(labels):
        if label in val_to_idx:
            result[array[..., idx]] = val_to_idx[label]
    
    return result

def rearrange_labels_dense(array, labels, mapping, default = 0, ** kwargs):
    one_hot = np.equal(
        np.expand_dims(array, axis = -1),
        np.arange(np.max(array) + 1).reshape([1] * len(array.shape) + [-1])
    )
    return rearrange_labels_one_hot(
        one_hot, labels, mapping, default, ** kwargs
    )

def rearrange_labels_sparse_one_hot(array, labels, mapping, default = 0, ** kwargs):
    last_index          = array.indices[:, -1]
    one_hot_last_index  = np.equal(
        np.expand_dims(last_index.numpy(), axis = -1),
        np.expand_dims(np.arange(array.dense_shape[-1]), axis = 0)
    )
    new_last_index      = tf.cast(rearrange_labels_one_hot(
        one_hot_last_index, labels, mapping, default, ** kwargs
    ), array.indices.dtype)
    
    new_indices = tf.concat([
        array.indices[:, :-1], tf.expand_dims(new_last_index, axis = -1)
    ], axis = -1)
    
    # filters out labels that do not have any mapping
    mask    = new_last_index != default
    new_indices = tf.boolean_mask(new_indices, mask)
    new_values  = tf.boolean_mask(array.values, mask)
    
    return tf.sparse.reorder(tf.sparse.SparseTensor(
        indices = new_indices,
        values  = new_values,
        dense_shape = tuple(array.dense_shape[:-1]) + (len(mapping), )
    ))

def rearrange_labels_sparse(array, labels, mapping, default = 0, ** kwargs):
    one_hot_values = np.equal(
        np.expand_dims(array.values.numpy(), axis = -1),
        np.expand_dims(np.arange(tf.reduce_max(array.values) + 1), axis = 0)
    )
    new_values  = rearrange_labels_one_hot(
        one_hot_values, labels, mapping, default, ** kwargs
    )
    # filters out labels that do not have any mapping
    mask    = new_values != default
    new_values  = new_values[mask]
    new_indices = tf.boolean_mask(array.indices, tf.cast(mask, tf.bool))
    
    return tf.sparse.reorder(tf.sparse.SparseTensor(
        indices = new_indices, values = new_values, dense_shape = array.dense_shape
    ))

def rearrange_labels_dense_slow(array, labels, mapping, default = 0, ** kwargs):
    uniques, indexes = np.unique(array, return_inverse = True)
    indexes = indexes.reshape(array.shape)
    
    unique_labels = {labels[idx] : idx for idx in uniques}
    
    val_to_idx = {}
    for i, l in enumerate(mapping):
        if not isinstance(l, (list, tuple)): l = [l]
        for l_i in l: val_to_idx[l_i] = i
    
    for label, idx in unique_labels.items():
        if val_to_idx.get(label, default) != idx:
            array[indexes == idx] = val_to_idx.get(label, default)
    return array
