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
import tensorflow as tf

from utils.med_utils.label_utils import rearrange_labels, build_mapping, transform_mask
from utils.med_utils.resampling import resample_volume
from utils.med_utils.pre_processing import get_frames

def get_nb_frames(data):
    if isinstance(data, dict):
        if 'nb_images' in data: return data['nb_images']
        if 'nb_frames' in data: return data['nb_frames']
        data = data['images']
    
    if not isinstance(data, (list, str)): raise ValueError('Invalid data (type {}) : {}'.format(type(data), data))
    
    if isinstance(data, list): return len(data)

    if os.path.isdir(data): return len(os.listdir(data))
    if data.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        
        return nib.load(data).shape[2]
    return 1
    
def load_medical_data(filename, ** kwargs):
    assert isinstance(filename, str) and os.path.isfile(filename)
    
    ext = max(_loading_fn.keys(), key = lambda ext: -1 if not filename.endswith(ext) else len(ext))

    if not filename.endswith(ext):
        raise ValueError('Unsupported file type : {}'.format(os.path.basename(filename)))

    return _loading_fn[ext](filename, ** kwargs)

def load_medical_image(filename,
                       voxel_dims   = None,
                       
                       target_shape     = None,
                       target_voxel_dims    = None,
                       
                       start_frame = -1,
                       end_frame   = -1,
                       
                       ** kwargs
                      ):
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
        image, voxel_dims = resample_volume(
            image,
            voxel_dims,
            target_shape    = target_shape,
            target_voxel_dims   = target_voxel_dims,
            ** kwargs
        )
    else:
        image = image.astype(np.float32)
    
    if start_frame != -1 or end_frame != -1:
        image = get_frames(image, start_frame, end_frame)
    
    return image, np.array(voxel_dims, dtype = np.float32)

def load_medical_seg(filename,
                     voxel_dims  = None,
                     mask_labels = None,
                     mapping     = None,
                     
                     target_shape      = None,
                     target_voxel_dims = None,
                     
                     start_frame    = -1,
                     end_frame      = -1,
                     
                     output_mode    = None,
                     
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
            - mapping     : the expected labels to keep in the mask (see `help(rearrange_labels)`)
            
            - target_{shape / voxel_dims} : arguments for the resizing method
            
            - kwargs : propagated to the resizing method
    """
    mask = filename
    if isinstance(filename, str):
        if os.path.isfile(filename):
            return_label    = mask_labels is None and mapping is not None
            mask_infos      = load_medical_data(filename, return_label = return_label)
            
            mask, mask_labels, voxel_dims = mask_infos if return_label else (
                mask_infos[0], mask_labels, mask_infos[1]
            )
        elif os.path.isdir(filename):
            filename = [os.path.join(filename, f) for f in os.listdir(filename)]
        else:
            raise ValueError('{} does not exist !'.format(filename))
    
    # if `filename` is a mapping {label : mask_file}
    if isinstance(filename, dict):
        # builds the label-mapping
        val_to_idx = build_mapping(labels = None, groups = mapping)
        
        mask = None
        for i, (label, file) in enumerate(filename.items()):
            if val_to_idx and l not in val_to_idx: continue
            
            m, voxel_dims = load_medical_data(file)
            if mask is None: mask = np.zeros(m.shape, dtype = np.uint8)
            mask[m.astype(bool)] = val_to_idx.get(label, i + 1)
    elif mapping:
        assert mask_labels is not None, 'When providing `mapping`, you must provide `mask_labels` !'
        
        mask = rearrange_labels(mask, mask_labels, mapping = mapping, default = 0)
    
    if mask is None:
        raise RuntimeError('`mask is None`, meaning that either `filename` is invalid (expected filename or dict {label : file}), or no label is valid')
    
    if target_shape is not None or target_voxel_dims is not None:
        assert voxel_dims is not None, 'You must provide `voxel_dims` when passing raw mask !\n{} !'.format(filename)
        
        #if len(mask.shape) == 4 and max_label_reshape > 0 and max_label_reshape < mask.shape[-1]:
        #    masks, new_voxel_dim = [], None
        #    for i in range(0, mask.shape[-1], max_label_reshape):
        #        print('Index {}'.format(i))
        #        m, vox = resample_volume(
        #            tf.cast(mask[..., i : i + max_label_reshape], tf.uint8), voxel_dims, target_voxel_dims = target_voxel_dims, target_shape = target_shape, ** kwargs
        #        )
        #        masks.append(m.numpy())
        #        new_voxel_dim = vox
        #        del m
            
        #    mask, voxel_dims = np.concatenate(masks, axis = -1), new_voxel_dim
        
        mask, voxel_dims = resample_volume(
            mask,
            voxel_dims,
            target_shape    = target_shape,
            target_voxel_dims   = target_voxel_dims,
            ** kwargs
        )

    if start_frame != -1 or end_frame != -1:
        mask = get_frames(mask, start_frame, end_frame)

    if output_mode:
        is_one_hot = False
        if mapping and mask.shape[-1] == len(mapping): is_one_hot = True
        if mask_labels and mask.shape[-1] == len(mask_labels): is_one_hot = True
        
        n_labels = -1
        if mapping:         n_labels = len(mapping)
        elif mask_labels:   n_labels = len(mask_labels)
        
        mask = transform_mask(mask, output_mode, is_one_hot, n_labels )

    return mask, np.array(voxel_dims, dtype = np.float32)

def _nibabel_load(filename, return_label = False):
    import nibabel as nib
    
    data    = nib.load(filename)
    mask    = data.get_fdata(caching = 'unchanged')
    pixdims = data.header['pixdim'][1:4]
    labels  = None
    if return_label:
        if not data.extra or 'labels' not in data.extra:
            raise RuntimeError('When `return_label = True`, the Nifti file should have an `extra` field with the `labels` key, which is not the case for {}'.format(filename))
        labels = data.extra['labels']
    
    return (mask, pixdims) if not return_label else (mask, labels, pixdims)

def _numpy_load(filename, return_label = False):
    with np.load(filename) as file:
        mask, voxel_dims = tf.sparse.SparseTensor(
            indices     = file['mask'],
            values      = tf.ones((len(file['mask']), ), dtype = tf.uint8),
            dense_shape = file['shape']
        ), file['pixdim']
    
    return (mask, voxel_dims) if not return_label else (mask, None, voxel_dims)

_loading_fn = {
    'nii'    : _nibabel_load,
    'nii.gz' : _nibabel_load,
    'npz'    : _numpy_load
}