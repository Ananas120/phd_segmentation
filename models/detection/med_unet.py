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
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.ndimage.filters import gaussian_filter

from loggers import timer
from utils import convert_to_str
from models.interfaces.base_image_model import BaseImageModel
from utils.med_utils import load_medical_data, save_medical_data, build_mapping, crop_then_reshape, pad_or_crop

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

def _get_step_size(image_shape, patch_size, step_size):
    patch_size     = np.array(patch_size)
    image_shape    = np.array(image_shape)
    
    step_in_voxels = patch_size * step_size
    
    num_steps = np.ceil((image_shape - patch_size) / step_in_voxels).astype(np.int32) + 1
    
    steps = []
    for shape_i, patch_i, steps_i in zip(image_shape, patch_size, num_steps):
        if steps_i > 1:
            max_step = shape_i - patch_i
            actual_step_size = max_step / (steps_i - 1)
            
            steps.append(np.round(np.arange(steps_i) * actual_step_size).astype(np.int32))
        else:
            steps.append([0])
        
    
    return steps

def _get_gaussian(patch_size, sigma_scale = 1. / 8.):
    patch_size = np.array(patch_size)
    
    center = patch_size // 2
    sigmas = patch_size * sigma_scale
    
    gaussian = np.zeros(patch_size)
    gaussian[tuple(center)] = 1
    gaussian = gaussian_filter(gaussian, sigmas, 0., mode = 'constant', cval = 0)
    gaussian = gaussian / np.max(gaussian)
    
    mask = gaussian == 0
    gaussian[mask] = np.min(gaussian[~mask])
    
    return gaussian.astype(np.float32)

def _parse_data(data, key, get_frames = True, ** kwargs):
    filename, start, end = data, -1, -1
    if isinstance(data, (dict, pd.Series)):
        if get_frames:
            if 'start_frame' in data: start = data['start_frame']
            if 'end_frame' in data:   end = data['end_frame']
        filename = data[key]
    return filename, start, end

def build_normalization_with_clip(mean, sd, percentile_00_5, percentile_99_5, ** kwargs):
    def normalize(data):
        data = tf.clip_by_value(data, percentile_00_5, percentile_99_5)
        return (data - mean) / sd
    
    mean = np.reshape(mean, [1, 1, 1, -1])
    sd   = np.reshape(sd, [1, 1, 1, -1])
    percentile_00_5 = np.reshape(percentile_00_5, [1, 1, 1, -1])
    percentile_99_5 = np.reshape(percentile_99_5, [1, 1, 1, -1])
    return normalize

class MedUNet(BaseImageModel):
    def __init__(self,
                 labels,
                 
                 input_size    = (None, None, 1),
                 voxel_dims    = None,
                 resize_method = 'pad',
                 
                 n_frames   = 1,
                 slice_axis = 2,
                 transpose  = None,
                 
                 nb_class      = None,
                 pad_value     = None,
                 
                 mapping       = None,
                 
                 image_normalization = None,
                 
                 pretrained         = None,
                 pretrained_task    = None,
                 pretrained_task_id = None,

                 ** kwargs
                ):
        if pretrained or pretrained_task or pretrained_task_id:
            from custom_architectures.totalsegmentator_arch import get_nnunet_plans, get_totalsegmentator_model_infos
            
            pretrained, infos = get_totalsegmentator_model_infos(pretrained, task = pretrained_task, task_id = pretrained_task_id)
            
            plans = get_nnunet_plans(model_name = pretrained)
            
            n_frames   = -1
            voxel_dims = plans['plans_per_stage'][0]['current_spacing']
            transpose  = [2, 1, 0]
            if labels is not None and not labels:
                if 'classes' in infos:
                    labels = infos['classes']
                else:
                    labels = [None] + plans['all_classes']
            
            image_normalization = plans['dataset_properties']['intensityproperties'][0]
            kwargs.update({
                'pretrained'      : pretrained,
                'pretrained_name' : pretrained,
                'resize_kwargs'   : {'interpolation' : 'bicubic', 'preserve_aspect_ratio' : False}
            })
        
        if isinstance(image_normalization, dict):
            kwargs['image_normalization_fn'] = build_normalization_with_clip(** image_normalization)
        
        self._init_image(
            input_size = input_size, resize_method = resize_method, image_normalization = image_normalization, ** kwargs
        )
        self.voxel_dims = voxel_dims
        
        self.n_frames   = max(n_frames, 1) if n_frames is not None and n_frames >= 0 else None
        self.slice_axis = slice_axis
        self.transpose  = transpose
        self.pad_value  = pad_value
        
        if labels is None and nb_class is None:
            self.labels   = None
            self.nb_class = None
        else:
            self.labels   = list(labels) if not isinstance(labels, str) else [labels]
            self.nb_class = max(max(1, nb_class if nb_class is not None else 1), len(self.labels))
            if self.nb_class > len(self.labels):
                self.labels += [''] * (self.nb_class - len(self.labels))
        
        self.mapping     = mapping
        self._tf_mapping = build_mapping(mapping if mapping else self.labels, output_format = 'tensor')
        
        super().__init__(** kwargs)

    def _build_model(self, final_activation, architecture = 'totalsegmentator', ** kwargs):
        super()._build_model(model = {
            'architecture_name' : architecture,
            'input_shape'       : tuple([s if s is None or s > 0 else None for s in self.input_shape]),
            'output_dim'        : self.last_dim,
            'final_activation'  : final_activation,
            ** kwargs
        })
    
    def _maybe_set_skip_empty(self, val, name):
        if val is None or not hasattr(self.get_loss(), name): return None
        getattr(self.get_loss(), name).assign(val)
    
    @property
    def is_3d(self):
        return False if self.n_frames == 1 else True

    @property
    def input_shape(self):
        return self.input_size if not self.is_3d else self.input_size[:2] + (self.n_frames, self.input_size[2])
    
    @property
    def last_dim(self):
        if self.nb_class is None:
            raise NotImplementedError('If `self.nb_class is None`, you must redefine this property')
        return self.nb_class
    
    @property
    def last_output_dim(self):
        return self.last_dim
    
    @property
    def input_signature(self):
        return tf.TensorSpec(
            shape = (None, ) + self.input_shape, dtype = tf.float32
        )

    @property
    def output_signature(self):
        return tf.SparseTensorSpec(
            shape = (None, ) + self.input_shape[:-1] + (self.last_output_dim, ), dtype = tf.uint8
        )

    @property
    def training_hparams(self):
        additional = {}
        if self.is_3d and self.n_frames in (-1, None):
            additional['max_frames'] = -1
        
        return super().training_hparams(
            ** self.training_hparams_image,
            crop_mode  = ['center', 'center', 'random'],
            skip_empty_frames = False,
            skip_empty_labels = True,
            ** additional
        )
    
    @property
    def training_hparams_mapper(self):
        mapper = super().training_hparams_mapper
        mapper.update({
            'skip_empty_frames'    : lambda v: self._maybe_set_skip_empty(v, 'skip_empty_frames'),
            'skip_empty_labels'    : lambda v: self._maybe_set_skip_empty(v, 'skip_empty_labels')
        })
        return mapper

    def __str__(self):
        des = super().__str__()
        des += self._str_image()
        if self.labels is not None:
            des += "- Labels (n = {}) : {}\n".format(len(self.labels), self.labels)
        if self.voxel_dims is not None:
            des += "- Voxel dims : {}\n".format(self.voxel_dims)
        if self.is_3d:
            des += "- # frames (axis = {}) : {}\n".format(self.slice_axis, self.n_frames if self.n_frames else 'variable')
        return des
    
    @timer
    def infer(self, data, win_len = -1, hop_len = -1, use_argmax = False, ** kwargs):
        def _remove_padding(output):
            output = output[0, : data.shape[0], : data.shape[1], : data.shape[2]]
            if self.transpose:
                transpose_fn = np.transpose if isinstance(output, np.ndarray) else tf.transpose
                output = transpose_fn(output, self.transpose + ([3] if not use_argmax else []))
            return output
        
        if not self.has_variable_input_size: win_len = self.input_shape
        if not isinstance(win_len, int): win_len = tuple(win_len)
        
        unbatched_rank = 4 if self.is_3d else 3
        if len(data.shape) == unbatched_rank + 1: data = data[0]
        
        volume = self.preprocess_input(data) if not isinstance(win_len, tuple) else data
        volume = tf.expand_dims(volume, axis = 0)

        if not self.is_3d:
            if isinstance(win_len, tuple): raise ValueError('Model is 2D and therefore does not support inference with patch !')
            return _remove_padding(self._infer_2d(
                volume, win_len = win_len, use_argmax = use_argmax, ** kwargs
            ))
        
        if self.n_frames not in (-1, None): win_len = self.n_frames
        if win_len == -1: win_len = self.max_frames if self.max_frames not in (None, -1) else tf.shape(volume)[-2]
        if hop_len == -1: hop_len = win_len
        
        if isinstance(win_len, tuple):
            pred = self._infer_with_patch(
                volume, win_len, use_argmax = use_argmax, ** kwargs
            )
        elif win_len > 0 and win_len < volume.shape[-2]:
            infer_fn = self._infer_with_overlap if hop_len != win_len else self._infer_without_overlap
            
            pred = infer_fn(
                volume, win_len = win_len, hop_len = hop_len, use_argmax = use_argmax, ** kwargs
            )
        else:
            pred = self(volume, training = False)
            if use_argmax: pred = tf.argmax(pred, axis = -1, output_type = tf.int32)
        
        return _remove_padding(pred)

    def _infer_with_patch(self, volume, patch_size, step_size = 0.5, use_argmax = False, ** kwargs):
        steps = _get_step_size(volume.shape[1:-1], patch_size, step_size)

        if tf.reduce_any(tf.shape(volume)[1 : -1] < tf.cast(patch_size, tf.int32)):
            volume = tf.pad(volume, [
                (0, 0), * [(0, max(0, patch_size[i] - volume.shape[i + 1])) for i in range(len(patch_size))], (0, 0)
            ])
        logger.info('volume shape : {} - steps : {}'.format(volume.shape, steps))

        gaussian = _get_gaussian(patch_size, 1. / 8.)
        gaussian = np.reshape(gaussian, [1, * gaussian.shape, 1])

        pred   = np.zeros(volume.shape[:-1] + [self.last_dim], dtype = np.float32)
        counts = np.zeros(volume.shape, dtype = np.float32)
        for start_x in steps[0]:
            for start_y in steps[1]:
                for start_z in steps[2]:
                    pred_patch = self(volume[
                        :, # batch axis
                        start_x : start_x + patch_size[0],
                        start_y : start_y + patch_size[1],
                        start_z : start_z + patch_size[2]
                    ], training = False)
                    if use_argmax: pred_patch = tf.nn.softmax(pred_patch, axis = -1)
                    
                    pred_patch = pred_patch.numpy() * gaussian

                    pred[
                        :,
                        start_x : start_x + patch_size[0],
                        start_y : start_y + patch_size[1],
                        start_z : start_z + patch_size[2]
                    ] += pred_patch

                    counts[
                        :,
                        start_x : start_x + patch_size[0],
                        start_y : start_y + patch_size[1],
                        start_z : start_z + patch_size[2]
                    ] += gaussian

        pred = (pred / counts).astype(np.float32)
        return pred if not use_argmax else np.argmax(pred, axis = -1).astype(np.int32)

    def _infer_with_overlap(self, volume, win_len = -1, hop_len = -1, use_argmax = False, ** kwargs):
        n_slices = tf.cast(tf.math.ceil((tf.shape(volume)[-2] - win_len + 1) / hop_len), tf.int32)
        
        pad = n_slices * hop_len + win_len - volume.shape[-2]
        if pad > 0:
            n_slices += 1
            volume = tf.pad(volume, [(0, 0), (0, 0), (0, 0), (0, pad), (0, 0)])
        
        pred  = np.zeros(tuple(volume.shape)[:-1] + (self.last_dim, ), dtype = np.float32)
        count = np.zeros((1, 1, 1, volume.shape[3], 1), dtype = np.int32)
            
        for i in tf.range(n_slices):
            time_logger.start_timer('prediction')
            out_i = self(
                volume[..., i * hop_len : i * hop_len + win_len, :], training = False
            )
            time_logger.stop_timer('prediction')
            
            time_logger.start_timer('post-processing')
            pred[..., i * hop_len : i * hop_len + win_len, :]  += out_i.numpy()
            count[..., i * hop_len : i * hop_len + win_len, :] += 1
            time_logger.stop_timer('post-processing')

        pred = pred / count
        return pred if not use_argmax else np.argmax(pred, axis = -1).astype(np.int32)
    
    @tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
    def _infer_without_overlap(self,
                               volume  : tf.Tensor,
                               win_len : tf.Tensor,
                               hop_len : tf.Tensor,
                               use_argmax = False,
                               ** kwargs
                              ):
        n_slices = tf.cast(tf.math.ceil(tf.shape(volume)[-2] / win_len), tf.int32)
        
        pad = n_slices * win_len - tf.shape(volume)[-2]
        if pad > 0:
            volume = tf.pad(volume, [(0, 0), (0, 0), (0, 0), (0, pad), (0, 0)])
        
        pred     = tf.TensorArray(
            dtype = tf.float32 if not use_argmax else tf.int32, size = n_slices
        )
        for i in tf.range(n_slices):
            out_i = tf.transpose(self(
                volume[..., i * hop_len : i * hop_len + win_len, :], training = False
            ), [3, 0, 1, 2, 4])
            if use_argmax: out_i = tf.argmax(out_i, axis = -1, output_type = tf.int32)
            pred = pred.write(i, out_i)
        
        perms = [1, 2, 3, 0, 4] if not use_argmax else [1, 2, 3, 0]
        return tf.transpose(pred.concat(), perms)

    @tf.function(reduce_retracing = True, experimental_follow_type_hints = True)
    def _infer_2d(self, volume : tf.Tensor, win_len : tf.Tensor, use_argmax = False, ** kwargs):
        if self.slice_axis != 0:
            perms  = [1, 0, 2, 3] if self.slice_axis == 1 else [2, 0, 1, 3]
            volume = tf.transpose(volume, perms)
        
        n_slices = tf.cast(tf.math.ceil(tf.shape(volume)[0] / win_len), tf.int32)

        pred     = tf.TensorArray(
            dtype = tf.float32 if not use_argmax else tf.int32, size = tf.shape(volume)[0]
        )
        for i in tf.range(n_slices):
            out_i = self(
                volume[i * win_len : i * win_len + win_len], training = False
            )
            if use_argmax: out_i = tf.argmax(out_i, axis = -1, output_type = tf.int32)
            pred = pred.write(i, out_i)
        
        pred = pred.concat()
        if self.slice_axis != 0:
            perms  = [1, 0, 2] if self.slice_axis == 1 else [1, 2, 0]
            if not use_argmax: perms = perms + [3]
            volume = tf.transpose(volume, perms)
        return pred

    def preprocess_image(self, image, voxel_dims, mask = None, resize_to_multiple = True, ** kwargs):
        if tf.reduce_any(tf.shape(image) == 0):
            return image if mask is None else (image, mask)
        
        if self.is_3d:
            tar_shape = (self.input_size[0], self.input_size[1], self.n_frames)
            max_shape = (self.max_image_shape + (self.max_frames, )) if self.max_image_size not in (-1, None) else (-1, -1, self.max_frames)
        else:
            tar_shape = self.input_size[:2]
            max_shape = self.max_image_shape

        tar_shape = [s if s is not None and  s > 0 else -1 for s in tar_shape]
        max_shape = [s if s is not None and  s > 0 else -1 for s in max_shape]

        #tf.print('target_shape', target_shape)
        normalized = crop_then_reshape(
            image,
            mask = mask,
            voxel_dims   = voxel_dims,
            target_voxel_dims = self.voxel_dims,
            
            max_shape      = max_shape,
            target_shape   = tar_shape,
            multiple_shape = self.downsampling_factor if self.resize_method == 'resize' else None,
            
            crop_mode    = self.crop_mode,
            pad_value    = self.pad_value,
            pad_mode     = 'after',
            ** self.resize_kwargs
        )
        if mask is not None:
            normalized, mask = normalized
            if isinstance(self.output_signature, tf.SparseTensorSpec):
                if not isinstance(mask, tf.sparse.SparseTensor): mask = tf.sparse.from_dense(mask)
            elif isinstance(mask, tf.sparse.SparseTensor):
                mask = tf.sparse.to_dense(mask)

        normalized = self.normalize_image(normalized)
        #tf.print('normalized shape :', tf.shape(normalized))

        return (normalized, mask) if mask is not None else normalized

    def get_input(self, data, normalize = True, ** kwargs):
        filename, start, end = _parse_data(data, key = 'images', ** kwargs)
        
        image, voxel_dims = load_medical_data(
            filename,
            
            slice_start = start,
            slice_end   = end,
            slice_axis  = self.slice_axis,
            
            use_sparse  = False,
            dtype       = tf.float32
        )
        image = tf.ensure_shape(image, (None, None) if not self.is_3d else (None, None, None))
        
        if self.transpose is not None:
            image = tf.transpose(image, self.transpose)
        
        if self.input_size[-1] == 1:
            image = tf.expand_dims(image, axis = -1)
        
        if normalize: image = self.preprocess_image(image, voxel_dims, ** kwargs)
        
        return image if normalize else (image, voxel_dims)
    
    def get_output(self, data, ** kwargs):
        filename, start, end = _parse_data(data, key = 'segmentation', ** kwargs)
        
        mask, voxel_dims = load_medical_data(
            filename,
            
            slice_start = start,
            slice_end   = end,
            slice_axis  = self.slice_axis,
            
            labels      = data['label'],
            mapping     = self._tf_mapping,
            
            use_sparse  = True,
            is_one_hot  = True,
            
            dtype       = tf.uint8
        )
        mask.indices.set_shape([None, 3 if not self.is_3d else 4])
        mask.dense_shape.set_shape([3 if not self.is_3d else 4])

        if self.transpose is not None:
            mask = tf.sparse.transpose(mask, self.transpose + [3])
        return mask

    def filter_input(self, image):
        return tf.reduce_all(tf.shape(image) > 0)
    
    def filter_output(self, output):
        return True if not isinstance(output, tf.sparse.SparseTensor) else tf.shape(output.indices)[0] > 0
    
    def augment_input(self, image, ** kwargs):
        return self.augment_image(image, clip = False, ** kwargs)
    
    def preprocess_input(self, image, mask = None, ** kwargs):
        if self.resize_method != 'pad': return (image, mask) if mask is not None else image
        shape     = tf.shape(image)
        multiples = tf.cast(self.downsampling_factor, shape.dtype)
        shape     = shape[- len(multiples) - 1 : -1]
        
        tar_shape = tf.where(shape % multiples != 0, (shape // multiples + 1) * multiples, shape)
        return pad_or_crop(image, tar_shape, mask = mask, pad_mode = 'after', pad_value = self.pad_value)

    def encode_data(self, data):
        image, voxel_dims = self.get_input(data, normalize = False)
        mask              = self.get_output(data)
        
        return self.preprocess_image(image, voxel_dims, mask = mask)
    
    def filter_data(self, image, output):
        return tf.logical_and(self.filter_input(image), self.filter_output(output))

    def augment_data(self, image, output):
        return self.augment_input(image), output
    
    def preprocess_data(self, image, output):
        return self.preprocess_input(image, mask = output)
    
    def get_dataset_config(self, * args, ** kwargs):
        if not self.is_3d:
            kwargs.update({
                'padded_batch'     : True if self.has_variable_input_size else False,
                'pad_kwargs'       : {'padding_values' : (self.pad_value, 0)}
            })
        return super().get_dataset_config(* args, ** kwargs)
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            'voxel_dims' : self.voxel_dims,
            
            'n_frames'   : self.n_frames,
            'slice_axis' : self.slice_axis,
            'transpose'  : self.transpose,
            'pad_value'  : self.pad_value,
            
            'labels'     : self.labels,
            'mapping'    : self.mapping,
            'nb_class'   : self.nb_class
        })
        return config
