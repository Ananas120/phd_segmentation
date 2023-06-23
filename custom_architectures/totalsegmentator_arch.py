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
import tensorflow as tf
import tensorflow_addons as tfa

from pathlib import Path

from custom_layers import get_activation
from custom_architectures.current_blocks import _get_var, _get_concat_layer

logger = logging.getLogger(__name__)

TOTALSEGMENTATOR_LABELS = [
    None, 'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver', 'stomach', 'aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left', 'lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1', 'esophagus', 'trachea', 'heart_myocardium', 'heart_atrium_left', 'heart_ventricle_left', 'heart_atrium_right', 'heart_ventricle_right', 'pulmonary_artery', 'brain', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right', 'small_bowel', 'duodenum', 'colon', 'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12', 'humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'clavicula_left', 'clavicula_right', 'femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum', 'face', 'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right', 'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right', 'urinary_bladder'
]

TOTALSEGMENTATOR_URL    = 'http://94.16.105.223/static'
TOTALSEGMENTATOR_MODELS = {
    # the 5 parts of the full resolution model
    'Task251_TotalSegmentator_part1_organs_1139subj' : {
        'task_id' : 251, 'task' : 'organs', 'url' : '{base_url}/{name}.zip', 'classes' : [
            None, 'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver', 'stomach', 'aorta', 'inferior_vena_cava',
            'portal_vein_and_splenic_vein', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left', 'lung_upper_lobe_left',
            'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right'
        ]
    },
    'Task252_TotalSegmentator_part2_vertebrae_1139subj' : {
        'task_id' : 252, 'task' : 'vertebrae', 'url' : '{base_url}/{name}.zip', 'classes' : [
            None, 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11',
            'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4',
            'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6', 'vertebrae_C5', 'vertebrae_C4',
            'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1'
        ]
    },
    'Task253_TotalSegmentator_part3_cardiac_1139subj' : {
        'task_id' : 253, 'task' : 'cardiac', 'url' : '{base_url}/{name}.zip', 'classes' : [
            None, 'esophagus', 'trachea', 'heart_myocardium', 'heart_atrium_left', 'heart_ventricle_left', 'heart_atrium_right',
            'heart_ventricle_right', 'pulmonary_artery', 'brain', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left',
            'iliac_vena_right', 'small_bowel', 'duodenum', 'colon', 'urinary_bladder', 'face'
        ]
    },
    'Task254_TotalSegmentator_part4_muscles_1139subj' : {
        'task_id' : 254, 'task' : 'muscles', 'url' : '{base_url}/{name}.zip', 'classes' : [
            None, 'humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'clavicula_left', 'clavicula_right',
            'femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum', 'gluteus_maximus_left', 'gluteus_maximus_right',
            'gluteus_medius_left', 'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right',
            'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right'
        ]
    },
    'Task255_TotalSegmentator_part5_ribs_1139subj' : {
        'task_id' : 255, 'task' : 'ribs', 'url' : '{base_url}/{name}.zip', 'classes' : [
            None, 'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8',
            'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4',
            'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12'
        ]
    },
    # low resolution model (all classes at once)
    'Task256_TotalSegmentator_3mm_1139subj' : {
        'task_id' : 256, 'task' : 'total', 'url' : '{base_url}/{name}.zip'
    },
    # custom models from authors
    'Task269_Body_extrem_6mm_1200subj' : {
        'task_id' : 256, 'task' : 'body_extrem', 'url' : '{base_url}/{name}.zip'
    },    
    # custom models from constributor ?
    'Task258_lung_vessels_248subj' : {
        'task_id' : 258, 'task' : 'lung_vessels', 'url' : 'https://zenodo.org/record/7234263/files/{name}.zip?download=1'
    },
    'Task150_icb_v0' : {
        'task_id' : 150, 'task' : 'icb', 'url' : 'https://zenodo.org/record/7079161/files/{name}.zip?download=1'
    },
    'Task260_hip_implant_71subj' : {
        'task_id' : 260, 'task' : 'hip_implant', 'url' : 'https://zenodo.org/record/7234263/files/{name}.zip?download=1'
    },
    'Task503_cardiac_motion'     : {
        'task_id' : 503, 'task' : 'cardiac_motion', 'url' : 'https://zenodo.org/record/7271576/files/{name}.zip?download=1'
    },
    'Task273_Body_extrem_1259subj'     : {
        'task_id' : 273, 'task' : 'body_extrem', 'url' : 'https://zenodo.org/record/7510286/files/{name}.zip?download=1'
    },
    'Task315_thoraxCT'     : {
        'task_id' : 315, 'task' : 'thorax_ct', 'url' : 'https://zenodo.org/record/7510288/files/{name}.zip?download=1'
    },
    'Task008_HepaticVessel'     : {
        'task_id' : 8, 'task' : 'hepatic_vessel', 'url' : 'https://zenodo.org/record/7573746/files/{name}.zip?download=1'
    }
}


def l2_normalize(x, axis = -1):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis = axis)

def ConvBlock(x,
              filters,
              kernel_size = 3,
              strides     = 1,
              padding     = 'same',
              
              epsilon   = 1e-5,
              norm_type = 'instance',
              drop_rate = 0.25,
              
              duplicate_i    = True,
              manual_padding = True,
              
              n    = 2,
              name = None
             ):
    if norm_type == 'instance':
        norm_layer = tfa.layers.InstanceNormalization
    elif norm_type == 'layer':
        norm_layer = tf.keras.layers.LayerNormalization
    elif norm_type == 'batch':
        norm_layer = tf.keras.layers.BatchNormalization
    else:
        norm_layer = None
    
    for i in range(n):
        padding_i = padding
        
        if not isinstance(kernel_size, (tuple, np.ndarray)): kernel_size = (kernel_size, kernel_size, kernel_size)
        kernel_size = np.array(kernel_size)
        if np.any(kernel_size >= 3) and padding == 'same' and manual_padding:
            padding_i = 'valid'
            pad = kernel_size // 2
            x = tf.keras.layers.ZeroPadding3D(((pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])))(x)
        
        if duplicate_i:
            conv_name = '{}/{}/blocks/{}/conv'.format(name, i, i) if name else None
            norm_name = '{}/{}/blocks/{}/norm'.format(name, i, i) if name else None
        else:
            conv_name = '{}/blocks/{}/conv'.format(name, i, i) if name else None
            norm_name = '{}/blocks/{}/norm'.format(name, i, i) if name else None
        
        x = tf.keras.layers.Conv3D(filters, kernel_size, strides = strides if i == 0 else 1, padding = padding_i, name = conv_name)(x)
        if norm_layer is not None: x = norm_layer(epsilon = epsilon, name = norm_name)(x)
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        
        if drop_rate:
            x = tf.keras.layers.Dropout(drop_rate)(x)

    return x

def TotalSegmentator(input_shape    = (None, None, None, 1),
                     output_dim     = None,
                     normalize_output = False,
                     norm_type      = 'instance',
         
                     n_stages       = 6,
                     n_conv_per_stage   = 2,
                     filters        = [32, 64, 128, 256, 320, 320],
                     kernel_size    = 3,
                     strides        = [1, 2, 2, 2, 2, 1],
                     drop_rate      = 0.25,
                     manual_padding = True,
         
                     upsampling_activation  = None,
         
                     concat_mode    = 'concat',
                     
                     final_name     = 'seg_outputs',
                     final_activation   = 'sigmoid',
                     return_all_outputs = False,
      
                     mixed_precision     = False,
         
                     pretrained         = None,
                     pretrained_task    = None,
                     pretrained_task_id = None,
                     pretrained_ckpt    = 'model_final_checkpoint',
                     transfer_kwargs = {},
       
                     name   = None,
                     ** kwargs
                    ):
    if not isinstance(input_shape, tuple): input_shape = (input_shape, input_shape, 3)

    if pretrained or pretrained_task or pretrained_task_id:
        infos = get_nnunet_plans(model_name = pretrained, task = pretrained_task, task_id = pretrained_task_id)
        
        n_stages    = len(infos['plans_per_stage'][0]['conv_kernel_sizes'])
        kernel_size = [tuple(sizes) for sizes in infos['plans_per_stage'][0]['conv_kernel_sizes']]
        strides     = [tuple(sizes) for sizes in infos['plans_per_stage'][0]['pool_op_kernel_sizes']]
        strides     = [1] * (n_stages - len(strides)) + strides
        n_conv_per_stage = infos['conv_per_stage']
        if not output_dim: output_dim = infos['num_classes'] + 1
    
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    inputs  = tf.keras.layers.Input(shape = input_shape, name = 'input_image')

    x = inputs
    residuals = []
    ##############################
    #     Downsampling part      #
    ##############################
    
    for i in range(n_stages):
        x = ConvBlock(
            x,
            n           = _get_var(n_conv_per_stage, i),
            filters     = _get_var(filters, i),
            kernel_size = _get_var(kernel_size, i),
            strides     = _get_var(strides, i),
            norm_type   = _get_var(norm_type, i),
            drop_rate   = _get_var(drop_rate, i),
            name        = 'conv_blocks_context/{}'.format(i),
            duplicate_i = False if i < n_stages - 1 else True,
            manual_padding = manual_padding
        )
        if i < n_stages - 1:
            residuals.append(x)

    ####################
    # Upsampling part  #
    ####################
    
    outputs = []
    for i in range(n_stages - 1):
        x = tf.keras.layers.Conv3DTranspose(
            filters     = _get_var(filters, n_stages - i - 2),
            kernel_size = _get_var(strides, n_stages - i - 1),
            use_bias    = False,
            strides     = _get_var(strides, n_stages - i - 1),
            padding     = 'same',
            activation  = _get_var(upsampling_activation, i),
            name        = 'tu/{}'.format(i)
        )(x)

        concat_mode_i = _get_var(concat_mode, i)
        if concat_mode_i:
            x = _get_concat_layer(concat_mode_i)([x, residuals[- (i + 1)]])
        
        x = ConvBlock(
            x,
            n           = _get_var(n_conv_per_stage, n_stages - i - 2),
            filters     = _get_var(filters, n_stages - i - 2),
            kernel_size = _get_var(kernel_size, n_stages - i - 2),
            strides     = 1,
            norm_type   = _get_var(norm_type, n_stages - i - 2),
            manual_padding = manual_padding,
            drop_rate   = _get_var(drop_rate, i),
            name        = 'conv_blocks_localization/{}'.format(i)
        )
    
        if return_all_outputs or i == n_stages - 2:
            out = tf.keras.layers.Conv3D(
                output_dim, kernel_size = 1, strides = 1, use_bias = False, name = '{}/{}'.format(final_name, i)
            )(x)
            if normalize_output:
                out = tf.keras.layers.Lambda(
                    lambda x: l2_normalize(x, axis = -1)
                )(out)
            if final_activation:
                out = get_activation(final_activation, dtype = 'float32')(out)
            outputs.append(out)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs[::-1], name = name)
    
    if pretrained or pretrained_task or pretrained_task_id:
        from models.weights_converter import name_based_partial_transfer_learning
        
        state_dict = load_nnunet_model(
            model_name = pretrained, task = pretrained_task, task_id = pretrained_task_id, ckpt = pretrained_ckpt
        )

        name_based_partial_transfer_learning(model, state_dict, ** transfer_kwargs)
    
    return model

def get_totalsegmentator_model_name(model_name = None, task_id = None, task = None):
    """ Returns the key in TOTALSEGMENTATOR_MODELS for the given task / task_id """
    def _assert_available(value, availables, label):
        if value not in availables:
            raise ValueError('Unknown {} !\n  Accepted : {}\n  Got : {}'.format(
                label, tuple(availables.keys(), value)
            ))

    assert model_name or task or task_id is not None
    
    if model_name is not None:
        _assert_available(model_name, TOTALSEGMENTATOR_MODELS, 'model')
        return model_name
    
    if task is not None:
        task_to_model = {infos['task'] : name for name, infos in TOTALSEGMENTATOR_MODELS.items()}
        _assert_available(task, task_to_model, 'task')
        return task_to_model[task]
    
    if task is not None:
        task_id_to_model = {infos['task_id'] : name for name, infos in TOTALSEGMENTATOR_MODELS.items()}
        _assert_available(task_id, task_id_to_model, 'task')
        return task_id_to_model[task_id]

def get_totalsegmentator_model_infos(model_name = None, ** kwargs):
    model_name = get_totalsegmentator_model_name(model_name, ** kwargs)
    
    return model_name, TOTALSEGMENTATOR_MODELS[model_name]

def get_nnunet_plans(model_name = None, ** kwargs):
    from utils import load_data
    
    if not model_name: model_name = get_totalsegmentator_model_name(** kwargs)
    model_dir = download_nnunet(model_name = model_name)
    
    return load_data(os.path.join(model_dir, 'plans.pkl'))

def load_nnunet_model(model_name = None, ckpt = 'model_final_checkpoint', fold = None, ** kwargs):
    import torch
    
    if not model_name: model_name = get_totalsegmentator_model_name(** kwargs)
    model_dir = download_nnunet(model_name = model_name)
    
    if not fold:
        folds = [f for f in os.listdir(model_dir) if f.startswith('fold_')]
        logger.info('Available folds ({} used) : {}'.format(folds[0], folds))
        fold = folds[0]
    
    ckpt_file = os.path.join(model_dir, fold, ckpt + '.model')
    
    return torch.load(ckpt_file, map_location = 'cpu')['state_dict']

def download_nnunet(model_name = None, url = None):
    from models import _pretrained_models_folder

    if url is None:
        url = TOTALSEGMENTATOR_MODELS[model_name]['url']
    if model_name is None:
        model_name = os.path.basename(url).split('.')[0]
    
    if '{' in url: url = url.format(base_url = TOTALSEGMENTATOR_URL, name = model_name)
    base_dir = os.path.join(_pretrained_models_folder, 'pretrained_weights')
    model_dir = os.path.join(base_dir, model_name)
    
    if not os.path.exists(model_dir):
        import zipfile
        
        from utils import download_file
        
        logger.info('Downloading files for model {} at : {}'.format(model_name, url))
        filename = download_file(url, directory = base_dir)

        if filename is None:
            raise RuntimeError('Error while downloading (`filename is None`)')
        
        logger.info('Extracting files...')
        with zipfile.ZipFile(filename, 'r') as file:
            file.extractall(base_dir)
        
        #os.remove(filename)

    return os.path.join(model_dir, os.listdir(model_dir)[0])

custom_functions    = {
    'TotalSegmentator'    : TotalSegmentator
}
