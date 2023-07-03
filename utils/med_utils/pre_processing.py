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

from utils.med_utils.sparse_utils import sparse_pad
from utils.med_utils.resampling import _multiply, resample_volume, compute_new_shape

def extract_slices(data, start = -1, end = -1, axis = 2):
    if axis < 0: axis = len(tf.shape(data)) + axis
    if start == -1: start = 0
    
    if isinstance(data, (np.ndarray, tf.Tensor)):
        if end == -1: end = tf.shape(data)[axis]
        
        if axis == 0:   return data[start : end]
        elif axis == 1: return data[:, start : end]
        elif axis == 2: return data[:, :, start : end]
        else:           return data[..., start : end]

    elif isinstance(data, tf.sparse.SparseTensor):
        if end == -1: end = tf.shape(data)[axis]
        
        shape   = tf.shape(data)
        starts  = tf.concat([
            tf.zeros((axis, ), dtype = tf.int64),
            tf.cast(tf.fill((1, ), start), tf.int64),
            tf.zeros((len(shape) - axis - 1, ), dtype = tf.int64)
        ], axis = -1)
        lengths = tf.cast(tf.concat([
            shape[:axis],
            tf.reshape(tf.cast(end - start, shape.dtype), [-1]),
            shape[axis + 1 :]
        ], axis = -1), tf.int64)
        
        return tf.sparse.slice(data, starts, lengths)
    else:
        raise ValueError('Unknown data type ({}) : {}'.format(type(data), data))
        

def pad_or_crop(img,
                target_shape,
                mask    = None,
                
                crop_mode   = 'center',
                
                pad_mode    = 'after',
                pad_value   = None,
                ** kwargs
               ):
    def _get_start(diff, mode):
        if diff <= 1:          return 0
        elif mode == 'center': return diff // 2
        elif mode == 'random': return tf.random.uniform((), 0, diff, dtype = tf.int32)
        elif mode == 'random_center':    return tf.random.uniform((), diff // 4, (3 * diff) // 4, dtype = tf.int32)
        elif mode == 'random_center_20': return tf.random.uniform((), (2 * diff) // 5, (3 * diff) // 5, dtype = tf.int32)
        elif mode == 'random_center_80': return tf.random.uniform((), diff // 10, (9 * diff) // 10, dtype = tf.int32)
        elif mode == 'start':  return 0
        elif mode == 'end':    return diff
        else: return -1
    
    shape = tf.shape(img)
    diff  = shape[:len(target_shape)] - tf.cast(target_shape, shape.dtype)

    skip_mask = mask is None
    
    if tf.reduce_any(diff < 0):
        if pad_value is None: pad_value = tf.minimum(tf.cast(0, img.dtype), tf.reduce_min(img))
        pad = tf.maximum(- diff, 0)
        
        if pad_mode == 'before':
            padding = tf.concat([
                tf.expand_dims(pad, axis = 1),
                tf.zeros((tf.shape(pad)[0], 1), dtype = pad.dtype)
            ], axis = 1)
        elif pad_mode == 'after':
            padding = tf.concat([
                tf.zeros((tf.shape(pad)[0], 1), dtype = pad.dtype),
                tf.expand_dims(pad, axis = 1)
            ], axis = 1)
        else:
            pad_half = tf.expand_dims(pad // 2, axis = 1)
            padding  = tf.concat([
                pad_half, tf.expand_dims(pad, axis = 1) - pad_half
            ], axis = 1)
        
        if tf.shape(padding)[0] < len(tf.shape(img)):
            padding = tf.concat([
                padding, tf.zeros((len(tf.shape(img)) - tf.shape(padding)[0], 2), dtype = padding.dtype)
            ], axis = 0)
        
        img = tf.pad(
            img, padding, constant_values = pad_value
        )
        if not skip_mask:
            if tf.shape(padding)[0] < len(tf.shape(mask)):
                padding = tf.concat([
                    padding, tf.zeros((len(tf.shape(mask)) - tf.shape(padding)[0], 2), dtype = padding.dtype)
                ], axis = 0)
            
            if not isinstance(mask, tf.sparse.SparseTensor):
                mask = tf.pad(mask, padding)
            else:
                mask = sparse_pad(mask, padding)

    if tf.reduce_any(diff > 0):
        offsets = [_get_start(
            diff[i], crop_mode[i] if isinstance(crop_mode, (list, tuple)) else crop_mode
        ) for i in range(len(diff))]
        
        if len(shape) == 2:
            img = img[
                offsets[0] : offsets[0] + target_shape[0], offsets[1] : offsets[1] + target_shape[1]
            ]
        else:
            img = img[
                offsets[0] : offsets[0] + target_shape[0],
                offsets[1] : offsets[1] + target_shape[1],
                offsets[2] : offsets[2] + target_shape[2]
            ]
        
        if not skip_mask:
            if isinstance(mask, tf.sparse.SparseTensor):
                lengths = target_shape
                offsets = tf.cast(offsets, tf.int64)
                if len(tf.shape(mask)) > len(offsets):
                    offsets = tf.concat([offsets, tf.zeros((1, ), dtype = tf.int64)], axis = 0)
                    lengths = tf.concat([target_shape, [tf.shape(mask)[-1]]], axis = 0)
                
                mask = tf.sparse.slice(
                    mask, offsets, tf.cast(lengths, tf.int64)
                )
            else:
                if len(offsets) == 2:
                    mask = mask[
                        offsets[0] : offsets[0] + target_shape[0],
                        offsets[1] : offsets[1] + target_shape[1]
                    ]
                else:
                    mask = mask[
                        offsets[0] : offsets[0] + target_shape[0],
                        offsets[1] : offsets[1] + target_shape[1],
                        offsets[2] : offsets[2] + target_shape[2]
                    ]

    return img if skip_mask else (img, mask)

def crop_then_reshape(img,
                      voxel_dims,
                      target_shape,
                      target_voxel_dims,
                      
                      max_shape      = None,
                      multiple_shape = None,

                      mask = None,
                      interpolation = 'bilinear',
                      
                      ** kwargs
                     ):
    target_shape    = tf.cast(target_shape, tf.int32)
    max_inter_shape = tf.shape(img)[: len(target_shape)]
    
    factors      = compute_new_shape(
        target_shape, voxel_dims = target_voxel_dims, target_voxel_dims = voxel_dims, return_factors = True
    )
    if max_shape is not None:
        max_shape = tf.cast(max_shape, max_inter_shape.dtype)
        max_shape = tf.where(max_shape > 0, max_shape, max_inter_shape)
        max_inter_shape = tf.minimum(_multiply(max_shape, factors), max_inter_shape)

    intermediate_shape  = tf.where(
        target_shape > 0, _multiply(target_shape, factors), max_inter_shape
    )

    img = pad_or_crop(img, intermediate_shape, mask = mask, ** kwargs)
    if mask is not None: img, mask = img
    
    target_shape  = tf.where(target_shape > 0, target_shape, _multiply(tf.shape(img)[:len(target_shape)], 1. / factors))
    if max_shape is not None:
        target_shape = tf.minimum(target_shape, tf.cast(max_shape, target_shape.dtype))

    if multiple_shape is not None:
        multiple_shape = tf.cast(multiple_shape, target_shape.dtype)
        target_shape   = (target_shape // multiple_shape) * multiple_shape

    img, _ = resample_volume(
        img, voxel_dims, target_shape = target_shape, interpolation = interpolation, ** kwargs
    )
    if mask is None: return img
    
    return img, resample_volume(mask, voxel_dims, target_shape = target_shape, interpolation = 'nearest', ** kwargs)[0]
