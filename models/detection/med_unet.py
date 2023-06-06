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

from loggers import timer
from utils.med_utils import *
from utils import convert_to_str
from models.interfaces.base_image_model import BaseImageModel

def _parse_data(data):
    filename, start, end = data, -1, -1
    if isinstance(data, (dict, pd.Series)):
        if 'start_frame' in data: start = data['start_frame']
        if 'end_frame' in data:   end = data['end_frame']
        filename = data['filename'] if 'filename' in data else data['images']
    return filename, start, end
    
class MedUNet(BaseImageModel):
    def __init__(self,
                 labels,
                 input_size = (512, 512, 1),
                 voxel_dims = None,
                 n_frames   = 1,
                 
                 nb_class      = None,
                 pad_value     = None,
                 
                 mapping       = None,
                 
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
        
        self.mapping = build_mapping(self.labels, mapping)
        
        super().__init__(** kwargs)
    
    def get_output(self, data, ** kwargs):
        if self.nb_class == 1:
            shape = [None, None, None] if self.is_3d else [None, None]
        else:
            shape = [None, None, None, None] if self.is_3d else [None, None, None]

        mask = tf.py_function(
            self.get_output_fn, [data['segmentation'], data['label']], Tout = tf.SparseTensorSpec(shape = shape, dtype = tf.uint8)
        )
        mask.indices.set_shape([None, len(shape)])
        mask.values.set_shape([None])
        mask.dense_shape.set_shape([len(shape)])
        
        return mask

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
        return super().training_hparams(
            ** self.training_hparams_image,
            max_size   = (None, None),
            max_frames = -1,
            crop_mode  = ['center', 'center', 'random'],
            skip_empty_frames = False,
            skip_empty_labels = True
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

    def preprocess_image(self, image, voxel_dims, mask = None, resize_to_multiple = True, ** kwargs):
        if tf.reduce_any(tf.shape(image) == 0):
            return image if mask is None else (image, mask)
        
        if self.is_3d:
            tar_shape = (self.input_size[0], self.input_size[1], self.n_frames)
            max_shape = self.max_size + (self.max_frames, )
        else:
            tar_shape = self.input_size[:2]
            max_shape = self.max_size

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
            multiple_shape = self.downsampling_factor if resize_to_multiple else None,
            
            crop_mode    = self.crop_mode,
            pad_value    = self.pad_value
        )
        if mask is not None:
            normalized, mask = normalized
            if isinstance(self.output_signature, tf.SparseTensorSpec):
                if tf.reduce_any(tf.reduce_max(mask.indices, axis = 0) >= tf.expand_dims(tf.cast(tf.shape(mask), tf.int64), 0)):
                    tf.print(tf.shape(mask), tf.reduce_max(mask.indices, axis = 0), tar_shape, max_shape)
                if not isinstance(mask, tf.sparse.SparseTensor): mask = tf.sparse.from_dense(mask)
            elif isinstance(mask, tf.sparse.SparseTensor):
                mask = tf.sparse.to_dense(mask)

        normalized = self.normalize_image(normalized)
        #tf.print('normalized shape :', tf.shape(normalized))

        return (normalized, mask) if mask is not None else normalized
    
    def get_input_fn(self, filename, start_frame = -1, end_frame = -1, ** kwargs):
        return load_medical_image(
            convert_to_str(filename), start_frame = start_frame, end_frame = end_frame, ** kwargs
        )
    
    def get_output_fn(self, filename, file_labels = None, start_frame = -1, end_frame = -1, ** kwargs):
        if file_labels is not None: file_labels = convert_to_str(file_labels)
        filename = convert_to_str(filename)
        
        if file_labels and len(filename) == len(file_labels):
            filename = {label : file for label, file in zip(file_labels, filename)}
        
        return load_medical_seg(
            filename, mask_labels = file_labels, mapping = self.mapping,
            start_frame = start_frame, end_frame = end_frame, ** kwargs
        )[0]

    def get_input(self, data, normalize = True, ** kwargs):
        filename, start, end = _parse_data(data)
        
        image, voxel_dims = tf.numpy_function(
            self.get_input_fn, [filename], Tout = [tf.float32, tf.float32]
        )
        
        if self.input_size[-1] == 1:
            image.set_shape([None, None, None] if self.is_3d else [None, None])
            image = tf.expand_dims(image, axis = -1)
        else:
            image.set_shape([None, None, None, None] if self.is_3d else [None, None, None])
        
        voxel_dims.set_shape([3] if self.is_3d else [2])
        if normalize: image = self.preprocess_image(image, voxel_dims, ** kwargs)
        
        return image if normalize else (image, voxel_dims)
    
    def filter_input(self, image):
        return tf.reduce_all(tf.shape(image) > 0)
    
    def filter_output(self, output):
        return True if not isinstance(output, tf.sparse.SparseTensor) else tf.shape(output.indices)[0] > 0
    
    def augment_input(self, image, ** kwargs):
        return self.augment_image(image, clip = False)
    
    def encode_data(self, data):
        image, voxel_dims = self.get_input(data, normalize = False)
        mask              = self.get_output(data)
        
        return self.preprocess_image(image, voxel_dims, mask = mask)
    
    def filter_data(self, image, output):
        return tf.logical_and(self.filter_input(image), self.filter_output(output))

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
            'nb_class'   : self.nb_class
        })
        return config
