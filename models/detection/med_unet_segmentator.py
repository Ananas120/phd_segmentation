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

    @property
    def output_signature(self):
        return tf.SparseTensorSpec(
            shape = (None, ) + self.input_shape[:-1] + (self.last_dim, ), dtype = tf.uint8
        )
    
    def infer(self, images : tf.Tensor, win_len : tf.Tensor = -1, hop_len : tf.Tensor = -1):
        return np.argmax(super().infer(images, win_len, hop_len), axis = -1)
    
    def infer_old(self, data : tf.Tensor, win_len : tf.Tensor = -1, hop_len : tf.Tensor = -1):
        if self.is_3d:
            if win_len == -1: win_len = self.max_frames if self.max_frames is not None else tf.shape(images)[-2]
            if hop_len == -1: hop_len = win_len
        
        images = self.pad_to_multiple(data)
        if self.is_3d and win_len > 0 and win_len < tf.shape(images)[-2]:
            n_slices = tf.cast(tf.math.ceil((tf.shape(images)[-2] - win_len) / hop_len), tf.int32)
            
            pad = n_slices * hop_len + win_len - tf.shape(images)[-2]
            if pad > 0:
                n_slices += 1
                images = tf.pad(images, [(0, 0), (0, 0), (0, 0), (0, pad), (0, 0)])
            
            pred     = tf.TensorArray(
                dtype = tf.int32, size = n_slices
            )
            for i in tf.range(n_slices):
                out_i = tf.argmax(self(
                    images[..., i * hop_len : i * hop_len + win_len, :], training = False
                ), axis = -1, output_type = tf.int32)
                pred = pred.write(i, tf.transpose(out_i, [3, 0, 1, 2]))
                
            pred = tf.transpose(pred.concat(), [1, 2, 3, 0])
            if pad > 0: pred = pred[..., : - pad, :]
        else:
            pred = tf.argmax(self(images, training = False), axis = -1, output_type = tf.int32)
        
        return pred[:, : data.shape[1], : data.shape[2], : data.shape[3]]
    
    def compile(self, loss = 'DiceLoss', loss_config = {}, ** kwargs):
        loss_config.setdefault('from_logits', True)
        super().compile(loss = loss, loss_config = loss_config, ** kwargs)
    
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
    
