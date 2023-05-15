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

from models.detection.med_unet import MedUNet

class MedUNetClassifier(MedUNet):
    def _build_model(self, * args, final_activation = None, ** kwargs):
        if not final_activation:
            final_activation = 'sigmoid' if self.nb_class == 1 else 'softmax'
        
        super()._build_model(* args, final_activation = final_activation, ** kwargs)

    @property
    def output_signature(self):
        return tf.SparseTensorSpec(
            shape = (None, ) + self.input_shape[:-1], dtype = tf.int32
        )
    
    def compile(self, loss = 'sparse_categorical_crossentropy', ** kwargs):
        super().compile(loss = loss, ** kwargs)
    
    def get_output(self, data, ** kwargs):
        shape = [None, None, None] if self.is_3d else [None, None]

        mask = tf.numpy_function(
            self.get_output_fn, [data['segmentation'], data['label']], Tout = tf.int32
        )
        mask.set_shape([None, None, None] if self.is_3d else [None, None])
        
        return mask

