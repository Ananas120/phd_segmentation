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

from utils.generic_utils import convert_to_str
from utils.med_utils.sparse_utils import sparse_pad

TOTALSEGMENTATOR_LABELS = [
    None, 'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver', 'stomach', 'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left', 'lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1', 'esophagus', 'trachea', 'heart_myocardium', 'heart_atrium_left', 'heart_ventricle_left', 'heart_atrium_right', 'heart_ventricle_right', 'pulmonary_artery', 'brain', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right', 'small_bowel', 'duodenum', 'colon', 'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12', 'humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'clavicula_left', 'clavicula_right', 'femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum', 'face', 'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right', 'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right', 'urinary_bladder'
]

def add_zero_label(mask, is_one_hot):
    if not is_one_hot:
        if isinstance(mask, tf.sparse.SparseTensor):
            return tf.sparse.SparseTensor(
                indices = mask.indices, values = mask.values + 1, dense_shape = mask.dense_shape
            )
        else:
            raise NotImplementedError('Adding label is not supported for `Dense` tensor which is not one-hot encoded')
    elif isinstance(mask, np.ndarray):
        padding = [(0, 0)] * (len(mask.shape) - 1) + [(1, 0)]
        return np.pad(mask, padding)
    else:
        pad_fn = tf.pad if not isinstance(mask, tf.sparse.SparseTensor) else sparse_pad
        
        padding = tf.concat([
            tf.zeros((len(tf.shape(mask)) - 1, 2), dtype = tf.int32),
            tf.cast([[1, 0]], tf.int32)
        ], axis = 0)

        return pad_fn(mask, padding)


def build_mapping(labels, output_format = 'dict'):
    """
        Creates a mapping (dict) `{label : index}`
        
        Arguments :
            - labels : the (possibly nested) list of labels
                - dict : `{label : index}`, returned as is (already a mapping)
                - nested lists : (example [(l1, l2), l3] -> {l1 : 0, l2 : 0, l3 : 1})
                - list : [l1, l2, l3] -> {l1 : 0, l2 : 1, l3 : 2}
        Return :
            - dict : where keys are sub-labels, and values are the corresponding index
    """
    if labels is None: return {} if output_format == 'dict' else tf.cast([[]], tf.string)

    if output_format == 'dict':
        if isinstance(labels, dict): return labels

        mapping = {}
        for i, label in enumerate(convert_to_str(labels)):
            if not isinstance(label, (list, tuple)): label = [label]
            for l in label: mapping[l] = i
    
    elif output_format in ('tf', 'tensor'):
        if isinstance(labels, tf.Tensor):
            if len(tf.shape(labels)) == 1: labels = tf.expand_dims(labels, axis = 1)
            return labels
        
        max_len = max([len(l) if isinstance(l, (list, tuple)) else 0 for l in labels])
        
        if max_len == 0: return tf.expand_dims(tf.cast([l if l else '' for l in labels], tf.string), axis = 1)
        
        mapping = []
        for label in labels:
            if not label: label = ''
            if isinstance(label, str): label = [label]
            mapping.append(list(label) + [''] * (max_len - len(label)))
        mapping = tf.cast(mapping, tf.string)
    
    return mapping

def build_lookup_table(labels, mapping, default = 0):
    labels  = tf.cast(labels, tf.string)
    mapping = build_mapping(mapping, output_format  = 'tensor')
    
    match  = tf.reduce_any(labels[tf.newaxis, tf.newaxis, :] == mapping[:, :, tf.newaxis], axis = 1)
    
    mask   = tf.reduce_any(match, axis = 0)
    keys   = tf.boolean_mask(tf.range(tf.shape(labels)[0], dtype = tf.int32), mask)
    values = tf.boolean_mask(tf.argmax(match, axis = 0, output_type = tf.int32), mask)

    init  = tf.lookup.KeyValueTensorInitializer(keys, values)
    table = tf.lookup.StaticHashTable(init, default_value = default)
    return table

def rearrange_labels(array, labels, mapping, default = 0, is_one_hot = None, name = None, ** kwargs):
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
    with tf.name_scope(name or 'rearrange_labels'):
        if is_one_hot is None: is_one_hot  = array.shape[-1] == len(labels)
        is_sparse   = isinstance(array, tf.sparse.SparseTensor)

        tf_labels  = tf.cast(labels, tf.string)
        tf_mapping = build_mapping(mapping, output_format = 'tensor')
        if tf.reduce_all(tf_labels == tf_mapping[: tf.shape(tf_labels)[0], 0]):
            return array
        elif tf.reduce_all(tf_labels == tf_mapping[1 : tf.shape(tf_labels)[0] + 1, 0]):
            return add_zero_label(array, is_one_hot)
        else:
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

def rearrange_labels_one_hot(array, labels, mapping, default = 0, dtype = np.int32, ** kwargs):
    if isinstance(array, tf.Tensor):
        mapping   = build_mapping(mapping, output_format = 'tensor')
        new_depth = tf.shape(mapping)[0]
        return tf.one_hot(rearrange_labels_dense(
            tf.argmax(array, axis = -1), labels, mapping, default = default
        ), depth = new_depth, dtype = array.dtype)

    mapping   = build_mapping(mapping, output_format = 'dict')
    new_depth = max(mapping.values()) + 1

    array = array.astype(bool)

    result = np.full(array.shape[:-1] + (new_depth, ), default).astype(dtype)
    for idx, label in enumerate(convert_to_str(labels)):
        if label in mapping:
            result[..., mapping[label]] = np.logical_or(result[..., mapping[label]], array[..., idx])
    
    return result

def rearrange_labels_dense(array, labels, mapping, default = 0, ** kwargs):
    dtype = array.dtype
    
    table = build_lookup_table(labels, mapping, default = default)
    return tf.cast(table.lookup(tf.cast(array, tf.int32)), dtype)

def rearrange_labels_sparse_one_hot(array, labels, mapping, default = 0, ** kwargs):
    mapping = build_mapping(mapping, output_format = 'tensor')
    
    new_indices, new_values = array.indices, array.values
    new_shape = tf.concat([
        array.dense_shape[:-1], tf.cast(tf.reshape(tf.shape(mapping)[0], [-1]), tf.int64)
    ], axis = -1)

    if tf.shape(array.indices)[0] > 0:
        new_last_index = rearrange_labels_dense(array.indices[:, -1:], labels, mapping, default = default)
        new_last_index = tf.ensure_shape(new_last_index, (None, 1))
        
        new_indices = tf.concat([
            array.indices[:, :-1], new_last_index
        ], axis = -1)
        
        mask = new_last_index[:, 0] != default
        new_indices = tf.boolean_mask(new_indices, mask)
        new_values  = tf.boolean_mask(new_values, mask)
    
    return tf.sparse.reorder(tf.sparse.SparseTensor(
        indices = new_indices, values = new_values, dense_shape = new_shape
    ))

def rearrange_labels_sparse(array, labels, mapping, default = 0, ** kwargs):
    new_indices, new_values, new_shape = array.indices, array.values, array.dense_shape

    if tf.shape(array.indices)[0] > 0:
        new_values = rearrange_labels_dense(array.values, labels, mapping, default = default)
        
        mask = new_values != default
        new_indices = tf.boolean_mask(new_indices, mask)
        new_values  = tf.boolean_mask(new_values, mask)
    
    return tf.sparse.reorder(tf.sparse.SparseTensor(
        indices = new_indices, values = new_values, dense_shape = new_shape
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
