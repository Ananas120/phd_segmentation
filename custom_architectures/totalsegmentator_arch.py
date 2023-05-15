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
import tensorflow as tf
import tensorflow_addons as tfa

from pathlib import Path

from custom_layers import get_activation
from custom_architectures.current_blocks import _get_var, _get_concat_layer

def l2_normalize(x, axis = -1):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis = axis)

def ConvBlock(x, filters, kernel_size = 3, strides = 1, padding = 'same', n = 2, drop_rate = 0.25, name = None, duplicate_i = True, manual_padding = True):
    for i in range(n):
        padding_i = padding
        if kernel_size >= 3 and padding == 'same' and manual_padding:
            padding_i = 'valid'
            pad = kernel_size // 2
            x = tf.keras.layers.ZeroPadding3D(((pad, pad), (pad, pad), (pad, pad)))(x)
        
        if duplicate_i:
            conv_name = '{}/{}/blocks/{}/conv'.format(name, i, i) if name else None
            norm_name = '{}/{}/blocks/{}/norm'.format(name, i, i) if name else None
        else:
            conv_name = '{}/blocks/{}/conv'.format(name, i, i) if name else None
            norm_name = '{}/blocks/{}/norm'.format(name, i, i) if name else None
        
        x = tf.keras.layers.Conv3D(filters, kernel_size, strides = strides if i == 0 else 1, padding = padding_i, name = conv_name)(x)
        x = tfa.layers.InstanceNormalization(epsilon = 1e-5, name = norm_name)(x)
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        
        if drop_rate:
            x = tf.keras.layers.Dropout(drop_rate)(x)

    return x

def TotalSegmentator(input_shape    = (None, None, None, 1),
                     output_dim     = 104,
                     normalize_output = False,
         
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
         
                     pretrained = (256, 'Task256_TotalSegmentator_3mm_1139subj'),
                     transfer_kwargs = {},
       
                     name   = None,
                     ** kwargs
                    ):
    if not isinstance(input_shape, tuple): input_shape = (input_shape, input_shape, 3)

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
            kernel_size = 2 if i > 0 else 1,
            use_bias    = False,
            strides     = 2 if i > 0 else 1,
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
    
    if pretrained:
        import totalsegmentator.libs
        
        from nnunet.training.model_restore import load_model_and_checkpoint_files

        from models.weights_converter import name_based_partial_transfer_learning
        
        model_path = Path.home() /'.totalsegmentator/nnunet/results/nnUNet' / '3d_fullres' / pretrained[1]
        model_path = model_path / 'nnUNetTrainerV2_ep8000_nomirror__nnUNetPlansv2.1'

        totalsegmentator.libs.download_pretrained_weights(pretrained[0])

        ckpt = 'model_final_checkpoint'

        trainer, params = load_model_and_checkpoint_files(str(model_path), checkpoint_name = ckpt)
        trainer.load_checkpoint_ram(params[0], False)
        name_based_partial_transfer_learning(model, trainer.network.cpu(), ** transfer_kwargs)
    
    return model

custom_functions    = {
    'TotalSegmentator'    : TotalSegmentator
}
