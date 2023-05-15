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

from utils.med_utils.resampling import resample_volume, compute_new_shape

def pad_or_crop(img, target_shape, crop_mode = 'center', mask = None, pad_value = None, ** kwargs):
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
    diff  = shape - tf.cast(target_shape, shape.dtype)

    if tf.reduce_any(diff < 0):
        if pad_value is None: pad_value = tf.minimum(0., tf.reduce_min(img))
        pad = tf.maximum(- diff, 0)
        pad_half = pad // 2
        
        padding = [(pad_half[0], pad[0] - pad_half[0]), (pad_half[1], pad[1] - pad_half[1]), (pad_half[2], pad[2] - pad_half[2])]
        img = tf.pad(img, padding if len(shape) == 3 else padding + [(0, 0)], constant_values = pad_value)
        if mask is not None:
            if not isinstance(mask, tf.sparse.SparseTensor):
                mask = tf.pad(mask, padding if len(tf.shape(mask)) == 3 else padding + [(0, 0)])
            else:
                new_indices = mask.indices + tf.expand_dims(tf.cast(
                    [pad_half[0], pad_half[1], pad_half[2]] + ([0] if len(tf.shape(mask)) == 4 else []), mask.indices.dtype
                ), axis = 0)
                
                new_shape  = tf.shape(mask)[:3] + tf.cast(pad[:3], tf.shape(mask).dtype)
                if len(tf.shape(mask)) == 4:
                    new_shape = tf.concat([new_shape, [tf.shape(mask)[3]]], axis = 0)
                
                mask = tf.sparse.SparseTensor(
                    indices = new_indices,
                    values  = mask.values,
                    dense_shape = tf.cast(new_shape, tf.int64)
                )
    
    if tf.reduce_any(diff > 0):
        start_x = _get_start(diff[0] - 1, crop_mode[0] if isinstance(crop_mode, (list, tuple)) else crop_mode)
        start_y = _get_start(diff[1] - 1, crop_mode[1] if isinstance(crop_mode, (list, tuple)) else crop_mode)
        start_z = _get_start(diff[2] - 1, crop_mode[2] if isinstance(crop_mode, (list, tuple)) else crop_mode)
        
        img = img[start_x : start_x + target_shape[0], start_y : start_y + target_shape[1], start_z : start_z + target_shape[2]]
        if mask is not None:
            if isinstance(mask, tf.sparse.SparseTensor):
                mask = tf.sparse.slice(
                    mask,
                    [start_x, start_y, start_z] + ([0] if len(tf.shape(mask)) == 4 else []),
                    [target_shape[0], target_shape[1], target_shape[2]] + ([tf.shape(mask)[-1]] if len(tf.shape(mask)) == 4 else [])
                )
            else:
                mask = mask[start_x : start_x + target_shape[0], start_y : start_y + target_shape[1], start_z : start_z + target_shape[2]]

    return img if mask is None else (img, mask)

def crop_then_reshape(img, voxel_dims, target_shape, target_voxel_dims, mask = None, interpolation = 'bilinear', ** kwargs):
    intermediate_shape = compute_new_shape(target_shape, voxel_dims = target_voxel_dims, target_voxel_dims = voxel_dims)
    
    img = pad_or_crop(img, intermediate_shape, mask = mask, ** kwargs)
    if mask is not None: img, mask = img
    
    img, _ = resample_volume(img, voxel_dims, target_shape = target_shape, interpolation = interpolation, ** kwargs)
    if mask is None: return img
    
    return img, resample_volume(mask, voxel_dims, target_shape = target_shape, interpolation = 'nearest', ** kwargs)[0]
