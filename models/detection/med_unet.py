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
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import *

from loggers import timer
from utils.med_utils import *
from utils import convert_to_str
from models.interfaces.base_image_model import BaseImageModel

def pad_to_multiple(data, multiple, axis):
    if not isinstance(axis, (list, tuple)): axis = [axis]
    return tf.pad(
        data, [(0, (multiple - data.shape[i] % multiple) if i in axis else 0) for i in range(len(data.shape))]
    )

class MedUNet(BaseImageModel):
    def __init__(self,
                 labels,
                 input_size = (512, 512, 1),
                 voxel_dims = None,
                 n_frames   = 1,
                 
                 nb_class      = None,
                 pad_value     = None,
                 
                 ** kwargs
                ):
        self._init_image(input_size = input_size, ** kwargs)
        self.voxel_dims = voxel_dims
        
        self.n_frames   = max(n_frames, 1) if n_frames is not None and n_frames >= 0 else None
        self.pad_value  = pad_value
        
        if labels is None and nb_class is None:
            self.labels   = None
            self.nb_class = None
        else:
            self.labels   = list(labels) if not isinstance(labels, str) else [labels]
            self.nb_class = max(1, nb_class if nb_class is not None else len(self.labels))
            if self.nb_class > len(self.labels):
                self.labels += [''] * (self.nb_class - len(self.labels))

        self._downsampling_factor = -1
        
        super().__init__(** kwargs)

    @property
    def output_signature(self):
        raise NotImplementedError()
    
    def get_output(self, data, ** kwargs):
        raise NotImplementedError()

    def _build_model(self, final_activation, architecture = 'totalsegmentator', ** kwargs):
        super()._build_model(model = {
            'architecture_name' : architecture,
            'input_shape'       : self.input_shape,
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
    def downsampling_factor(self):
        if self._downsampling_factor == -1:
            downsampling_factor = 1
            for l in self.model.layers:
                if type(l) in (Conv2D, Conv3D, MaxPooling2D, MaxPooling3D):
                    downsampling_factor *= l.strides[0]
            self._downsampling_factor = downsampling_factor
        return self._downsampling_factor

    @property
    def input_shape(self):
        return self.input_size if not self.is_3d else self.input_size[:2] + (self.n_frames, self.input_size[2])
    
    @property
    def last_dim(self):
        if self.nb_class is None:
            raise NotImplementedError('If `self.nb_class is None`, you must redefine this property')
        return 1 if self.nb_class == 1 else self.nb_class
    
    @property
    def input_signature(self):
        return tf.TensorSpec(
            shape = (None, ) + self.input_shape, dtype = tf.float32
        )

    @property
    def training_hparams(self):
        return super().training_hparams(
            ** self.training_hparams_image,
            max_size   = (None, None),
            max_frames = -1,
            crop_mode  = ['center', 'center', 'random'],
            skip_empty_frames = False,
            skip_empty_labels = False
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
        if self.labels:
            des += "- Labels (n = {}) : {}\n".format(len(self.labels), self.labels)
        if self.voxel_dims:
            des += "- Voxel dims : {}\n".format(self.voxel_dims)
        if self.is_3d:
            des += "- # frames : {}\n".format(self.n_frames if self.n_frames else 'variable')
        return des
    
    def pad_to_multiple(self, data, axis = [1, 2]):
        return pad_to_multiple(data, self.downsampling_factor, axis)
        
    def infer(self, data : tf.Tensor, win_len : tf.Tensor = -1, hop_len : tf.Tensor = -1):
        if self.is_3d:
            if win_len == -1: win_len = self.max_frames if self.max_frames is not None else tf.shape(images)[-2]
            if hop_len == -1: hop_len = win_len
        
        images = self.pad_to_multiple(data)
        if self.is_3d and win_len > 0 and win_len < images.shape[-2]:
            n_slices = tf.cast(tf.math.ceil((tf.shape(images)[-2] - win_len) / hop_len), tf.int32)
            
            pad = n_slices * hop_len + win_len - images.shape[-2]
            if pad > 0:
                n_slices += 1
                images = tf.pad(images, [(0, 0), (0, 0), (0, 0), (0, pad), (0, 0)])
            
            pred  = np.zeros((images.shape[3], images.shape[0], images.shape[1], images.shape[2], self.last_dim), dtype = np.float32)
            count = np.zeros((images.shape[3], 1, 1, 1, 1), dtype = np.int32)
            
            for i in tf.range(n_slices):
                out_i = tf.transpose(self(
                    images[..., i * hop_len : i * hop_len + win_len, :], training = False
                ), [3, 0, 1, 2, 4])
                pred[i * hop_len : i * hop_len + win_len]  += out_i.numpy()
                count[i * hop_len : i * hop_len + win_len] += 1
            
            pred = np.transpose(pred / count, [1, 2, 3, 0, 4])
            if pad > 0: pred = pred[..., : - pad, :]
        else:
            pred = self(images, training = False).numpy()
        
        return pred[:, : data.shape[1], : data.shape[2]]

    def tf_infer(self, data : tf.Tensor, win_len : tf.Tensor = -1, hop_len : tf.Tensor = -1):
        if self.is_3d:
            if win_len == -1: win_len = self.max_frames if self.max_frames is not None else tf.shape(images)[-2]
            if hop_len == -1: hop_len = win_len
        
        images = self.pad_to_multiple(data)
        if self.is_3d and win_len > 0 and win_len < images.shape[-2]:
            n_slices = tf.cast(tf.math.ceil((tf.shape(images)[-2] - win_len) / hop_len), tf.int32)
            
            pad = n_slices * hop_len + win_len - images.shape[-2]
            if pad > 0:
                n_slices += 1
                images = tf.pad(images, [(0, 0), (0, 0), (0, 0), (0, pad), (0, 0)])
            
            pred     = tf.TensorArray(
                dtype = tf.float32, size = n_slices
            )
            for i in tf.range(n_slices):
                out_i = self(
                    images[..., i * hop_len : i * hop_len + win_len, :], training = False
                )
                pred = pred.write(i, tf.transpose(out_i, [3, 0, 1, 2, 4]))
            
            pred = tf.transpose(pred.concat(), [1, 2, 3, 0, 4])
            if pad > 0: pred = pred[..., : - pad, :]
        else:
            pred = self(images, training = False)
        
        return pred[:, : data.shape[1], : data.shape[2]]


    @timer(name = 'inference', log_if_root = False)
    def segment(self, image, training = False, ** kwargs):
        """
            Performs prediction on `image` and returns the model's output
            
            Arguments :
                - image : tf.Tensor of rank 3, 4 or 5 (single / batched image(s) / volume(s))
                - training  : whether to make prediction in training mode
            Return :
                - output : model's output of shape (B, ) + self.input_shape[:-1] + (self.nb_class, )
                
        """
        if not isinstance(image, tf.Tensor): image = tf.cast(image, tf.float32)
        if self.is_3d:
            if len(tf.shape(image)) == 4:
                image = tf.expand_dims(image, axis = 0)
        elif len(tf.shape(image)) == 3:
            image = tf.expand_dims(image, axis = 0)
        
        return self(image, training = training)

    def preprocess_image(self, image, voxel_dims, mask = None, ** kwargs):
        max_h = self.input_size[0] if self.input_size[0] is not None else self.max_size[0]
        max_w = self.input_size[1] if self.input_size[1] is not None else self.max_size[1]
        
        if max_h is None or max_h <= 0: max_h = tf.shape(image)[0]
        if max_w is None or max_w <= 0: max_w = tf.shape(image)[1]

        if self.is_3d:
            n_frames = self.n_frames if self.n_frames is not None else self.max_frames
            if n_frames <= 0: n_frames = tf.shape(image)[-2]
            target_shape = (
                max_h, max_w, n_frames, self.input_size[2]
            )
        else:
            target_shape = (max_h, max_w, self.input_size[2])
        
        #tf.print('target_shape', target_shape)
        normalized = crop_then_reshape(
            image, voxel_dims, target_shape = target_shape, target_voxel_dims = self.voxel_dims,
            mask = mask, crop_mode = self.crop_mode, pad_value = self.pad_value
        )
        if mask is not None:
            normalized, mask = normalized
            if isinstance(self.output_signature, tf.SparseTensorSpec):
                if tf.reduce_any(tf.reduce_max(mask.indices, axis = 0) >= tf.expand_dims(tf.cast(tf.shape(mask), tf.int64), 0)):
                    tf.print(tf.shape(mask), tf.reduce_max(mask.indices, axis = 0), target_shape)
                if not isinstance(mask, tf.sparse.SparseTensor): mask = tf.sparse.from_dense(mask)
            elif isinstance(mask, tf.sparse.SparseTensor):
                mask = tf.sparse.to_dense(mask)

        normalized = self.normalize_image(normalized)
        #tf.print('normalized shape :', tf.shape(normalized))

        return (normalized, mask) if mask is not None else normalized
    
    def get_input_fn(self, filename, ** kwargs):
        if isinstance(filename, np.ndarray) and filename.shape == (): filename = filename.item()
        if isinstance(filename, bytes): filename = filename.decode('utf-8')
        
        return load_medical_image(filename, ** kwargs)
    
    def get_output_fn(self, filename, file_labels = None, ** kwargs):
        if isinstance(filename, tf.Tensor): filename = filename.numpy()
        if isinstance(filename, np.ndarray) and filename.shape == (): filename = filename.item()
        if isinstance(filename, bytes): filename = filename.decode('utf-8')
        if file_labels is not None:
            if isinstance(file_labels, tf.Tensor): file_labels = file_labels.numpy()
            file_labels = [convert_to_str(label) for label in file_labels]
        
        if file_labels is not None and len(filename) == len(file_labels):
            filename = {label : convert_to_str(file) for label, file in zip(file_labels, filename)}
        
        return load_medical_seg(filename, mask_labels = file_labels, labels = self.labels, ** kwargs)[0]

    def get_input(self, data, normalize = True, ** kwargs):
        filename = data
        if isinstance(data, (dict, pd.Series)):
            filename = data['filename'] if 'filename' in data else data['images']
        
        image, voxel_dims = tf.numpy_function(
            self.get_input_fn, [filename], Tout = [tf.float32, tf.float32]
        )
        
        if self.input_size[-1] == 1:
            image.set_shape([None, None, None] if self.is_3d else [None, None])
            image = tf.expand_dims(image, axis = -1)
        else:
            image.set_shape([None, None, None, None] if self.is_3d else [None, None, None])
        
        voxel_dims.set_shape([3])
        if normalize: image = self.preprocess_image(image, voxel_dims)
        
        return image if normalize else (image, voxel_dims)
    
    def augment_input(self, image, ** kwargs):
        return self.augment_image(image, clip = False)
    
    def encode_data(self, data):
        (image, voxel_dims), mask = self.get_input(data, normalize = False), self.get_output(data)
        
        return self.preprocess_image(image, voxel_dims, mask = mask)
    
    def augment_data(self, image, output):
        return self.augment_input(image), output
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            'voxel_dims' : self.voxel_dims,
            'n_frames'   : self.n_frames,
            'pad_value'  : self.pad_value,
            'labels'     : self.labels,
            'nb_class'   : self.nb_class,
        })
        return config
