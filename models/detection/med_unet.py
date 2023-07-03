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

from loggers import timer
from models.interfaces.base_image_model import BaseImageModel
from utils import convert_to_str, create_iterator, load_json, dump_json, plot_volume
from utils.med_utils import (
    load_medical_data, save_medical_data, get_shape, build_mapping, crop_then_reshape, pad_or_crop, resample_volume, rearrange_labels
)

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

_gaussians = {}

def _get_step_size(image_shape, patch_size, step_size = 0.5):
    """
        Returns the start position for the patch-based slicing windows
        
        Arguments :
            - image_shape : (h, w, depth) of the image
            - patch_size  : (p_h, p_w, p_d), the size of the slicing windows in the 3-axes
            - step_size   : float, the fraction of the patch_size to use as step size
        Return :
            - start_positions : 3-items list where each item is a np.ndarray representing the start position in each axis
    """
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

@timer
def _get_gaussian(patch_size, sigma_scale = 1. / 8.):
    """ Return a 3-D gaussian (np.ndarray) of size `patch_size` """
    global _gaussians
    
    if tuple(patch_size) not in _gaussians:
        from scipy.ndimage.filters import gaussian_filter

        patch_size = np.array(patch_size)

        center = patch_size // 2
        sigmas = patch_size * sigma_scale

        gaussian = np.zeros(patch_size)
        gaussian[tuple(center)] = 1
        gaussian = gaussian_filter(gaussian, sigmas, 0., mode = 'constant', cval = 0)
        gaussian = gaussian / np.max(gaussian)

        mask = gaussian == 0
        gaussian[mask] = np.min(gaussian[~mask])
        
        _gaussians[tuple(patch_size)] = gaussian.astype(np.float32)

    return _gaussians[tuple(patch_size)]

def _parse_data(data, key, get_frames = True, ** kwargs):
    """ Return the filename (at `key` if `filename` is a `dict`) + possibly the start / end position (if provided) """
    filename, start, end = data, -1, -1
    if isinstance(data, (dict, pd.Series)):
        if get_frames:
            if 'start_frame' in data: start = data['start_frame']
            if 'end_frame' in data:   end = data['end_frame']
        filename = data[key]
    return filename, start, end

def build_normalization_with_clip(mean, sd, percentile_00_5, percentile_99_5, ** kwargs):
    """ Return a callable that normalizes the input by mean / sd after clipping it to the 5 / 95% percentiles (the arguments) """
    def normalize(data):
        data = tf.clip_by_value(data, percentile_00_5, percentile_99_5)
        return (data - mean) / sd
    
    mean = np.reshape(mean, [1, 1, 1, -1])
    sd   = np.reshape(sd, [1, 1, 1, -1])
    percentile_00_5 = np.reshape(percentile_00_5, [1, 1, 1, -1])
    percentile_99_5 = np.reshape(percentile_99_5, [1, 1, 1, -1])
    return normalize

@timer
def combine_preds(preds, models):
    """
        Combines a list of outputs
        
        Arguments :
            - preds  : list of `tf.Tensor / np.ndarray`, the model outputs to combine
            - models : list of `MedUNet` models, the models used (actually required to get their labels)
        Return :
            - combined : the combination of all `preds`
        
        Note : `0` is expected to be the "null" label (e.g. "background") for all models
        It means that the successive `preds[i]` will overwrite the result at every non-zero prediction of `preds[i]`
        The `models[i].labels` is used to give an offset at the prediction `preds[i]` to not have overlap between
        the different predicted labels
        
        Example : 
        ```
            preds    = [[0, 0, 2, 1], [1, 0, 1, 0]]
            combined = combine_preds(preds, ...) # labels = [[None, 'a', 'b'], [None, 'c']]
            print(combined) # [3, 0, 2, 3]
        ```
    """
    result = preds[0]
    if hasattr(result, 'numpy'): result = result.numpy()
    offset = len(models[0].labels) - 1
    for p, m in zip(preds[1:], models[1:]):
        if hasattr(p, 'numpy'): p = p.numpy()
        mask = p != 0
        result[mask] = p[mask] + offset
        offset += len(m.labels) - 1
    
    return result

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
        """
            Constructor for the base MedUNet interface
            
            Arguments :
                - labels : list of str, the list of labels (empty list to use the pretrained labels)
                           can also be `None` if the model is not a classifier
                
                - input_size : 3-element tuple (height, width, channel), the input size of the image
                               **IMPORTANT** in case of 3-D U-Net, the input size is also a 3-tuple (h, w, c)
                               The number of frames (or depth) is specified by the `n_frames` parameter
                - resize_method : whether to resize or pad the input for resizing to the expected `input_size`
                                  Note : the resizing for voxel resampling is **always** performed by interpolation and **not** padding
                                  This resizing is used to fit the `input_size` (if fixed) or resize to a multiple of the downscaling factor
                
                
                - n_frames   : the number of frames to use in case of 3-D UNet (`1` means 2-D UNet is used)
                - voxel_dims : 3-element tuple, the expected voxel dimensions in the 3 axes
                - slice_axis : on which axis to slice on
                - transpose  : the transposition axis for the input 3-D volume (used in `TotalSegmentator` pretrained models)
                
                - nb_class   : fix the expected number of labels (if > len(labels), labels is padded with empty labels)
                - pad_value  : the value used for padding
                
                - mapping    : a mapping for the labels (allows to combine multiple *mask labels* to one unique label
                               See the example in the notebook for more information on its usage
                
                - image_normalization : the image normalization scheme
                                        If `dict`, should contains the keys `mean`, `sd` and `percentile_{00 / 99}_5`
        """
        if pretrained or pretrained_task or pretrained_task_id:
            from custom_architectures.totalsegmentator_arch import get_nnunet_plans, get_totalsegmentator_model_infos
            
            pretrained, infos = get_totalsegmentator_model_infos(pretrained, task = pretrained_task, task_id = pretrained_task_id)
            
            plans = get_nnunet_plans(model_name = pretrained)
            
            n_frames   = -1
            if voxel_dims is None: voxel_dims = plans['plans_per_stage'][0]['current_spacing']
            if transpose is None:  transpose  = [2, 1, 0]
            if image_normalization is None:
                image_normalization = plans['dataset_properties']['intensityproperties'][0]

            if labels is not None and not labels:
                if 'classes' in infos:
                    labels = infos['classes']
                else:
                    labels = [None] + plans['all_classes']
            
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
            self.labels   = [str(l) if l is not None else l for l in self.labels]
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
        """
            Generic inference method that internally calls specific inference method :
            - if 2D UNet is used      : `_infer_2d` \*
            - if `win_len` is a tuple : `_infer_with_patch` \*\*
            - if `win_len <= hop_len` : `_infer_without_overlap` \*
            - if `win_len > hop_len`  : `_infer_with_overlap`
            - else : simple model call (also used if `win_len >= data.shape[self.slice_axis`)
            
            \* These functions are compiled in tensorflow graph, meaning that they are much faster than the other ones.
            \*\* This function is highly inspired from the `NNUNet` project
            
            Arguments :
                - data    : 4-D (h, w, d, c) volume (tf.Tensor / np.ndarray), the data to predict on
                - win_len : int or tuple (patch size), the slice size (if 2D, used as the batch_size)
                - hop_len : int, the step size (if `< win_len`, there is an overlap between slices)
                - use_argmax : bool, whether to perform an argmax or not (if `False`, outputs the logits)
                - kwargs     : forwarded to the specific inference function (mainly ignored)
            Return :
                - pred : 3-D int32 (labels, if `use_argmax == True`) or 4-D float32 (logits) tf.Tensor or np.ndarray, the output
            
        """
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
                    time_logger.start_timer('patch prediction')
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
                    time_logger.stop_timer('patch prediction')

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
        """ Main processing function that normalizes and resize `image` to the expected `self.voxel_dims` """
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
        """ Pads `image` (and `mask`) to a multiple of `self.downsampling_factor` (if `resize_method == 'pad'`) """
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
    
    @timer
    def post_process(self, pred, origin, labels = None, mapping = None, ** kwargs):
        dtype = tf.int32 if pred.dtype in (np.int32, tf.int32) else tf.float32
        if mapping and pred.dtype in (np.int32, tf.int32):
            if not labels: labels = self.labels
            pred = rearrange_labels(pred, labels = labels, mapping = mapping, is_one_hot = False)
        
        shape_origin = get_shape(origin)
        pred =  resample_volume(pred, voxel_dims = self.voxel_dims, target_shape = shape_origin, interpolation = 'nearest')[0]
        return tf.cast(pred, dtype)

    @timer
    def predict(self,
                filenames,
                
                friends = None,
                
                mapping = None,
                
                save = True,
                overwrite   = False,
                directory   = None,
                output_dir  = None,
                output_file = 'pred_{}.npz',
                
                display     = False,
                plot_kwargs = {'strides' : 3},
                
                verbose     = False,
                tqdm = lambda x: x,
                
                ** kwargs
               ):
        ##############################
        #      Utility functions     #
        ##############################
        
        def _get_filename(file):
            if isinstance(file, (dict, pd.Series)):
                file = file['images'] if 'images' in file else file['filename']
            return file
        
        ####################
        #  Initialization  #
        ####################
        
        time_logger.start_timer('initialization')
        
        if verbose or display: tqdm = lambda x: x
        if friends is not None and not isinstance(friends, (list, tuple)): friends = [friends]
        if not isinstance(filenames, (list, tuple, pd.DataFrame)): filenames = [filenames]
        filenames = [_get_filename(file) for file in create_iterator(filenames)]
        
        predicted = {}
        if save:
            if directory is None:  directory = self.pred_dir
            if output_dir is None: output_dir = os.path.join(directory, 'segmentations')
            
            os.makedirs(directory, exist_ok = True)
            os.makedirs(output_dir, exist_ok = True)
            
            map_file  = os.path.join(directory, 'map.json')
            predicted = load_json(map_file, default = {})
        
        to_predict = filenames if overwrite else [file for file in filenames if file not in predicted]
        to_predict = list(set(to_predict))
        
        time_logger.stop_timer('initialization')

        ########################################
        #     Dataset creation + main loop     #
        ########################################
        
        #dataset    = prepare_dataset(to_predict, map_fn = self.get_input, batch_size = 0, cache = False)
        all_labels = self.labels
        if friends:
            from models import get_pretrained
            friends = [get_pretrained(f) if isinstance(f, str) else f for f in friends]
            for f in friends: all_labels += f.labels[1:]
        labels     = all_labels if mapping is None else mapping
        
        for file in tqdm(to_predict):
            time_logger.start_timer('pre-processing')
            inp = self.get_input(file)
            time_logger.stop_timer('pre-processing')

            pred  = self.infer(inp, ** kwargs)
            
            if friends:
                time_logger.start_timer('friends prediction')
                
                preds = [pred]
                for f in friends:
                    if verbose: logger.info('Making prediction with friend {}...'.format(f.nom))
                    time_logger.start_timer('inference of {}'.format(f.nom))
                    
                    if f.image_normalization == self.image_normalization and f.voxel_dims == self.voxel_dims:
                        f_out = f.infer(inp, ** kwargs)
                    else:
                        f_out = f.infer(f.get_input(file), ** kwargs)
                    preds.append(f_out if not hasattr(f_out, 'numpy') else f_out.numpy())
                    
                    time_logger.stop_timer('inference of {}'.format(f.nom))
                
                pred = combine_preds(preds, models = [self] + friends)
                
                time_logger.stop_timer('friends prediction')
            
            pred = self.post_process(pred, file, labels = all_labels, mapping = mapping)
            
            if display:
                time_logger.start_timer('display')
                plot_volume(pred, labels = labels, ** plot_kwargs)
                time_logger.stop_timer('display')
            
            infos = {'segmentation' : pred, 'labels' : labels}
            if save:
                time_logger.start_timer('saving mask')
                
                out_file = predicted.get(file, {}).get('segmentation', output_file)
                if callable(out_file): out_file = out_file(file)
                elif '{}' in out_file: out_file = out_file.format(len(os.listdir(output_dir)))
                
                if file not in predicted: out_file = os.path.join(output_dir, out_file)
                
                save_medical_data(out_file, infos['segmentation'], origin = file, labels = labels, is_one_hot = False)
                
                infos['segmentation'] = out_file
                time_logger.stop_timer('saving mask')

            predicted[file] = infos
            
            if save:
                time_logger.start_timer('saving json')
                dump_json(map_file, predicted, indent = 4)
                time_logger.stop_timer('saving json')

        return [(file, predicted[file]) for file in filenames]

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
