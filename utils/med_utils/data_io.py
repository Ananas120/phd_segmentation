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

import os
import numpy as np
import nibabel as nib
import tensorflow as tf

from utils.med_utils import resampling

def load_medical_data(filename, ** kwargs):
    assert isinstance(filename, str) and os.path.isfile(filename)
    
    ext = max(_loading_fn.keys(), key = lambda ext: -1 if not filename.endswith(ext) else len(ext))

    if not filename.endswith(ext):
        raise ValueError('Unsupported file type : {}'.format(os.path.basename(filename)))

    return _loading_fn[ext](filename, ** kwargs)

def load_medical_image(filename, voxel_dims = None, target_voxel_dims = None, target_shape = None, ** kwargs):
    if isinstance(filename, tuple) and len(filename) == 2: filename, voxel_dims = filename
    
    image = filename
    if isinstance(filename, str):
        if os.path.isdir(filename):
            filename = [os.path.join(filename, f) for f in os.listdir(filename)]
        else:
            image, voxel_dims = load_medical_data(filename)
    
    if isinstance(filename, dict): filename = list(filename.values())
    if isinstance(filename, list):
        image, voxel_dims = list(zip(* [load_medical_image(f) for f in filename]))
        image, voxel_dims = np.stack(image, axis = 2), voxel_dims[0]
    
    if target_shape is not None or target_voxel_dims is not None:
        assert voxel_dims is not None, 'You must provide `voxel_dims` when passing raw image : {} !'.format(filename)
        image, voxel_dims = resampling.resample_volume(
            image, voxel_dims, target_voxel_dims = target_voxel_dims, target_shape = target_shape, ** kwargs
        )
    else:
        image = image.astype(np.float32)
    
    return image, np.array(voxel_dims, dtype = np.float32)

def load_medical_seg(filename,
                     voxel_dims  = None,
                     mask_labels = None,
                     labels      = None,
                     
                     target_shape      = None,
                     target_voxel_dims = None,
                     
                     remove_label_axe  = False,
                     max_label_reshape = -1,
                     
                     ** kwargs
                    ):
    """
        Loads and returns a segmentation mask, and possibly filters it for requested `labels` + reshaping it
        
        Arguments :
            - filename : the file(s) for the mask or the mask itself
                - str   : either nifti file, either RT-STRUCT
                - dict  : {label : mask_file}
                - np.ndarray : raw mask (you must provide `voxel_dims` and/or `mask_labels`)  if you want filtering / reshaping
            - voxel_dims  : the current mask voxel dims (inferred if `filename` is a (list of) file(s))
            - mask_labels : the mask labels (inferred if `filename` is a (list of) file(s))
            - labels      : the expected labels to keep in the mask
            
            - target_{shape / voxel_dims} : arguments for the resizing method
            
            - kwargs : propagated to the resizing method
    """
    mask = filename
    if isinstance(filename, str):
        if os.path.isfile(filename):
            out = load_medical_data(filename, return_label = True if mask_labels is None and labels else False)
            mask, mask_labels, voxel_dims = out if mask_labels is None and labels else (out[0], mask_labels, out[1])
        else:
            filename = [os.path.join(filename, f) for f in os.listdir(filename)]
    
    if isinstance(filename, dict):
        mask = None
        for i, (l, f) in enumerate(filename.items()):
            if labels and l not in labels: continue
            
            m, voxel_dims = load_medical_data(f)
            if mask is None: mask = np.zeros(m.shape, dtype = np.uint8)
            mask[np.where(m)] = labels.index(l) if labels else i + 1
    elif isinstance(mask, np.ndarray):
        if labels:
            assert mask_labels, 'You must provide `mask_labels` when passing raw mask for label filtering !'
            if labels[:len(mask_labels)] != mask_labels:
                new_order = [idx for idx, label in enumerate(mask_labels) if label in labels]
                mask      = mask[..., new_order]
    elif isinstance(mask, (tf.Tensor, tf.sparse.SparseTensor)):
        if labels:
            assert mask_labels, 'You must provide `mask_labels` when passing raw mask for label filtering !'
            if labels[0] is None and labels[1 : len(mask_labels) + 1] == mask_labels:
                if isinstance(mask, tf.sparse.SparseTensor):
                    mask = tf.sparse.SparseTensor(
                        indices = mask.indices + tf.cast([[0] * (mask.indices.shape[1] - 1) + [1]], mask.indices.dtype),
                        values  = mask.values,
                        dense_shape = tuple(mask.dense_shape[:-1]) + (len(labels), )
                    )
                else:
                    mask = tf.pad(mask, [(0, 0)] * (len(mask.indices.shape[1]) - 1) + [(1, 0)])
            elif labels[:len(mask_labels)] != mask_labels:
                raise NotImplementedError()
    else:
        raise NotImplementedError('Only dict `object : mask` if currently supported\n  Got (type {}) : {}'.format(type(filename), filename))
    
    if remove_label_axe:
        if isinstance(mask, np.ndarray) and len(mask.shape) == 4:
            mask = np.argmax(mask, axis = -1)
        elif isinstance(mask, tf.Tensor) and len(mask.shape) == 4:
            mask = tf.argmax(mask, axis = -1)
        elif isinstance(mask, tf.sparse.SparseTensor) and len(mask.dense_shape) == 4:
            mask = tf.argmax(tf.sparse.to_dense(mask), axis = -1)
    
    if isinstance(mask, np.ndarray) or mask.dtype != tf.uint8:
        mask = tf.cast(mask, tf.uint8)
    
    if target_shape is not None or target_voxel_dims is not None:
        assert voxel_dims is not None, 'You must provide `voxel_dims` when passing raw mask : {} !'.format(filename)
        if len(mask.shape) == 4 and max_label_reshape > 0 and max_label_reshape < mask.shape[-1]:
            masks, new_voxel_dim = [], None
            for i in range(0, mask.shape[-1], max_label_reshape):
                print('Index {}'.format(i))
                m, vox = resampling.resample_volume(
                    tf.cast(mask[..., i : i + max_label_reshape], tf.uint8), voxel_dims, target_voxel_dims = target_voxel_dims, target_shape = target_shape, ** kwargs
                )
                masks.append(m.numpy())
                new_voxel_dim = vox
                del m
            
            mask, voxel_dims = np.concatenate(masks, axis = -1), new_voxel_dim
        else:
            mask, voxel_dims = resampling.resample_volume(
                tf.cast(mask, tf.uint8), voxel_dims, target_voxel_dims = target_voxel_dims, target_shape = target_shape, ** kwargs
            )

    return mask, voxel_dims

def _nibabel_load(filename, return_label = False):
    data   = nib.load(filename)
    mask   = data.get_fdata(caching = 'unchanged')
    labels = None
    if return_label:
        if not data.extra or 'labels' not in data.extra:
            raise RuntimeError('When `return_label = True`, the Nifti file should have an `extra` field with the `labels` key, which is not the case for {}'.format(filename))
        labels = data.extra['labels']
    
    return (mask, data.header['pixdim'][1:4]) if not return_label else (mask, labels, data.header['pixdim'][1:4])

def _numpy_load(filename, return_label = False):
    with np.load(filename) as file:
        return tf.sparse.SparseTensor(
            indices     = file['mask'],
            values      = tf.ones((len(file['mask']), ), dtype = tf.uint8),
            dense_shape = file['shape']
        ), file['pixdim']

_loading_fn = {
    'nii'    : _nibabel_load,
    'nii.gz' : _nibabel_load,
    'npz'    : _numpy_load
}