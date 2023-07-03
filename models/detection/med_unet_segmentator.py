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

class MedUNetSegmentator(MedUNet):
    def __init__(self, * args, obj_threshold = 0.5, ** kwargs):
        self.obj_threshold  = obj_threshold

        super().__init__(** kwargs)
    
    def _build_model(self, * args, final_activation = None, ** kwargs):
        super()._build_model(* args, final_activation = final_activation, ** kwargs)
    
    def infer(self, * args, use_argmax = True, ** kwargs):
        return super().infer(* args, use_argmax = use_argmax, ** kwargs)
    
    def compile(self, loss = 'DiceLoss', loss_config = {}, ** kwargs):
        loss_config.setdefault('from_logits', True)
        super().compile(loss = loss, loss_config = loss_config, ** kwargs)
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            'obj_threshold' : self.obj_threshold
        })
        return config
    
