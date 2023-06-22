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
    
    def predict(self,
                images,
                max_frames = None,
                overlap    = None,
                
                save = True,
                directory  = None,
                output_dir = None,
                filename   = 'pred_{}.npy',
                
                tqdm = lambda x: x,
                
                ** kwargs
               ):
        if not isinstance(images, (list, tuple, pd.DataFrame)): images = [images]
        
        predicted = {}
        if save:
            if directory is None:  directory = self.pred_dir
            if output_dir is None: output_dir = os.path.join(directory, 'segmentations')
            
            os.makedirs(directory, exist_ok = True)
            os.makedirs(output_dir, exist_ok = True)
            
            map_file  = os.path.join(directory, 'map.json')
            predicted = load_json(map_file, default = {})
        
        results = []
        for path in images:
            if path in predicted and not overwrite:
                results.append((path, predicted[path]))
                continue
            
            image = tf.expand_dims(self.get_input(path), axis = 0)
            
            win_len = max_frames if isinstance(max_frames, int) else -1
            if isinstance(max_frames, float): win_len = int(max_frames / self.voxel_dims[-1])
            
            hop_len = overlap if isinstance(overlap, int) else -1
            if isinstance(overlap, float): hop_len = int(overlap / self.voxel_dims[-1])
            
            pred  = self.infer(image, win_len = win_len, hop_len = hop_len).numpy()
            
            infos = {'segmentation' : pred, 'labels' : self.labels}
            if save:
                file = predicged.get(path, {}).get('segmentation', None)
                if not file:
                    file = os.path.join(output_dir, file)
                    if '{}' in file: file = file.format(len(glob.glob(file.replace('{}', '*'))))
                
                if file.endswith('.npy'):
                    np.save(file, pred)
                elif file.endswith(('.nii', '.nii.gz')):
                    raise NotImplementedError('Nifti saving is not supported yet !')
                
                predicted[path] = {** infos, 'segmentation' : file}
                dump_json(map_file, predicted, indent = 4)
            
            results.append((path, infos))
        
        return results
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            'obj_threshold' : self.obj_threshold
        })
        return config
    
