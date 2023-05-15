
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils.image.geo_utils import *
from models.interfaces.base_image_model import BaseImageModel

logger  = logging.getLogger(__name__)

class EAST(BaseImageModel):
    def __init__(self,
                 labels = None,
                 nb_class   = None,
                 input_size = 512,

                 obj_threshold  = 0.35,

                 ** kwargs
                ):
        self._init_image(input_size = input_size, ** kwargs)

        if labels is None: labels = ['object']
        self.labels   = list(labels) if not isinstance(labels, str) else [labels]
        self.nb_class = max(1, nb_class if nb_class is not None else len(self.labels))
        if self.nb_class > len(self.labels):
            self.labels += [''] * (self.nb_class - len(self.labels))

        self.obj_threshold  = obj_threshold
        
        self.label_to_idx   = {label : i for i, label in enumerate(self.labels)}
        
        if self.use_labels: raise NotImplementedError('Not fully supported yet !')
        
        super().__init__(** kwargs)

    def _build_model(self, architecture = 'unet', ** kwargs):
        super()._build_model(model = {
            'architecture_name' : architecture,
            'input_shape'   : self.input_size,
            'output_dim'    : [1, 5] + ([self.nb_class] if self.use_labels else []),
            'final_activation'  : ['sigmoid', 'sigmoid', 'softmax'],
            'final_name'    : ['score_map', 'geo_map', 'class'],
            ** kwargs
        })
    
    @property
    def use_labels(self):
        return False if self.nb_class == 1 else True
    
    @property
    def output_signature(self):
        sign = (
            tf.TensorSpec(shape = (None, ) + self.input_size[:-1], dtype = tf.float32),
            tf.TensorSpec(shape = (None, ) + self.input_size[:-1] + (5, ), dtype = tf.float32),
            tf.TensorSpec(shape = (None, ) + self.input_size[:-1], dtype = tf.bool),
        )
        if self.use_labels:
            sign += (
                tf.TensorSpec(shape = (None, ) + self.input_size[:-1] + (1, ), dtype = tf.int32), 
            )
        return sign
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            ** self.training_hparams_image,
            min_poly_size   = 6,
            max_wh_factor   = 5,
            shrink_ratio    = 0.1
        )
    
    def __str__(self):
        des = super().__str__()
        des += self._str_image()
        des += "- Labels (n = {}) : {}\n".format(len(self.labels), self.labels)
        return des
    
    def compile(self, loss = 'EASTLoss', ** kwargs):
        super().compile(loss = loss, ** kwargs)

    def decode_output(self, model_output, ** kwargs):
        score_map = model_output[1][0, :, :, 0]
        geo_map   = model_output[0][0, :, :, :] * score_map[:, :, np.newaxis]

        # filter the score map
        points = np.argwhere(score_map > self.obj_threshold)

        # sort the text boxes via the y axis
        points = points[np.argsort(points[:, 0])]

        # restore
        return restore_rectangle_rbox(points[:, ::-1], geo_map[points[:, 0], points[:, 1]])

    def get_rbox(self,
                 polys,
                 labels,
                 img_size,
                 
                 min_poly_size  = -1,
                 shrink_ratio   = -1,
                 max_wh_factor  = -1
                ):
        """
            Generates score_map and geo_map

            Arguments :
                - polys : np.ndarray of shape [N, 4, 2] of `(y, x)` coordinates
                - tags  : np.ndarray of labels
                - im_size   : (height, width) of the image
                - min_poly_size : the minimal area for a polygon to be valid
            Returns :
                - score_map : np.ndarray of shape `im_size` with value of 1 for pixels within a box
                - geo_map   : np.ndarray of shape `im_size + (5, )` where the last axis represents
                    the distance between the top / right / bottom / left sides of the rectangle
                    and the rotation angle
        """
        if min_poly_size == -1: min_poly_size = self.min_poly_size
        if shrink_ratio == -1:  shrink_ratio = self.shrink_ratio
        if max_wh_factor == -1: max_wh_factor = self.max_wh_factor
        
        return get_rbox_map(
            polys,
            img_shape   = img_size,
            out_shape   = self.input_size[:2],
            labels      = labels,
            mapping     = self.label_to_idx if self.use_labels else None,
            
            min_poly_size   = min_poly_size,
            shrink_ratio    = shrink_ratio,
            max_wh_factor   = max_wh_factor
        )

    def get_input(self, filename, ** kwargs):
        return self.get_image(filename)
    
    def filter_data(self, inputs, outputs):
        if tf.reduce_any(tf.math.is_nan(outputs[1])): return False
        return tf.reduce_any(tf.logical_and(
            tf.cast(outputs[0], tf.bool), outputs[-1]
        ))
    
    def augment_input(self, image, ** kwargs):
        return self.augment_image(image)
    
    def preprocess_input(self, image, ** kwargs):
        return self.preprocess_image(image)
    
    def get_output(self, data):
        outputs = tf.numpy_function(
            self.get_rbox, [data['mask'], data['label'], (data['height'], data['width'])],
            Tout = [tf.float32, tf.float32, tf.bool, tf.int32]
        )
        score_map, geo_map, valid_mask = outputs[:3]
        
        score_map   .set_shape(self.input_size[:2])
        geo_map     .set_shape([self.input_size[0], self.input_size[1], 5])
        valid_mask  .set_shape(self.input_size[:2])
        
        return score_map, geo_map, valid_mask
    
    def encode_data(self, data):
        return self.get_input(data), self.get_output(data)
    
    def augment_data(self, image, output):
        return self.augment_input(image), output
    
    def preprocess_data(self, image, output):
        return self.preprocess_input(image), output
                                
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            'labels'    : self.labels,
            'nb_class'  : self.nb_class,
            'obj_threshold' : self.obj_threshold
        })
        
        return config

