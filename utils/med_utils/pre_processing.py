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

import tensorflow as tf

from utils.med_utils.resampling import _multiply, resample_volume, compute_new_shape

def get_frames(data, start = -1, end = -1):
    if start == -1: start = 0
    if isinstance(data, (np.ndarray, tf.Tensor)):
        if end == -1: end = data.shape[2]
        assert len(data.shape) > 2, 'Data must have at least 3 dimensions (height, width, frames[, channels])'
        return data[..., start : end] if len(data.shape) == 3 else data[..., start : end, :]
    elif isinstance(data, tf.sparse.SparseTensor):
        if end == -1: end = data.dense_shape[2]
        starts  = [0, 0, start]
        lengths = [data.dense_shape[0], data.dense_shape[1], end - start]
        if len(data.dense_shape) == 4:
            starts += [0]
            lengths += [data.dense_shape[3]]
        
        return tf.sparse.slice(data, starts, lengths)
    else:
        raise ValueError('Unknown data type ({}) : {}'.format(type(data), data))
        
    
def pad_or_crop(img,
                target_shape,
                mask    = None,
                
                crop_mode   = 'center',
                
                pad_mode    = 'even',
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

    if tf.reduce_any(diff < 0):
        if pad_value is None: pad_value = tf.minimum(tf.cast(0, img.dtype), tf.reduce_min(img))
        pad = tf.maximum(- diff, 0)
        
        if pad_mode == 'before':
            padding = [(pad[i], 0) for i in range(len(pad))]
        elif pad_mode == 'after':
            padding = [(0, pad[i]) for i in range(len(pad))]
        else:
            pad_half = pad // 2
            padding = [(pad_half[i], pad[i] - pad_half[i]) for i in range(len(pad))]
        
        img = tf.pad(
            img, padding if len(shape) == len(padding) else padding + [(0, 0)], constant_values = pad_value
        )
        if mask is not None:
            if not isinstance(mask, tf.sparse.SparseTensor):
                mask = tf.pad(mask, padding if len(tf.shape(mask)) <= 3 else padding + [(0, 0)])
            else:
                offset = [padding[i][0] for i in range(len(padding))]
                if len(offset) < len(tf.shape(mask)): offset = offset + [0]
                
                new_indices = mask.indices + tf.expand_dims(tf.cast(offset, mask.indices.dtype), axis = 0)
                
                new_shape   = tf.shape(mask)[: len(pad)]
                new_shape   = new_shape + tf.cast(pad, new_shape.dtype)
                if len(new_shape) < len(tf.shape(mask)):
                    new_shape = tf.concat([new_shape, tf.shape(mask)[len(pad) :]], axis = 0)
                
                mask = tf.sparse.SparseTensor(
                    indices = new_indices,
                    values  = mask.values,
                    dense_shape = tf.cast(new_shape, tf.int64)
                )
    
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
        
        if mask is not None:
            if isinstance(mask, tf.sparse.SparseTensor):
                mask = tf.sparse.slice(
                    mask,
                    offsets + ([0] if len(tf.shape(mask)) > len(offsets) else []),
                    [target_shape[0], target_shape[1], target_shape[2]] + ([tf.shape(mask)[-1]] if len(tf.shape(mask)) > len(offsets) else [])
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

    return img if mask is None else (img, mask)

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
