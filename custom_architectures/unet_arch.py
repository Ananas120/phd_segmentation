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

import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D

from custom_architectures.current_blocks import (
    _get_var, _get_concat_layer, Conv2DBN, Conv3DBN, Conv2DTransposeBN, Conv3DTransposeBN
)

def UNet(input_shape    = 512,
         output_dim     = 1,
         
         n_stages       = 5,
         n_conv_per_stage   = lambda i: 1 if i == 0 else 2,
         
         filters        = [32, 64, 128, 256, 512],
         kernel_size    = 3,
         strides        = 1,
         use_bias       = True,
         
         activation     = 'relu',
         pool_type      = 'max',
         pool_strides   = 2,
         bnorm          = 'never',
         drop_rate      = 0.3,
         
         upsampling_activation  = None,
         
         concat_mode    = 'concat',
         
         final_name     = 'segmentation_layer',
         final_activation   = 'sigmoid',
         
         name   = None,
         ** kwargs
        ):
    if not isinstance(input_shape, tuple): input_shape = (input_shape, input_shape, 3)

    if len(input_shape) == 3:
        conv_fn     = Conv2DBN
        pool_fn     = MaxPooling2D if pool_type == 'max' else AveragePooling2D
        upsample_fn = Conv2DTransposeBN
    else:
        conv_fn     = Conv3DBN
        pool_fn     = MaxPooling3D if pool_type == 'max' else AveragePooling3D
        upsample_fn = Conv3DTransposeBN
    
    inputs  = tf.keras.layers.Input(shape = input_shape, name = 'input_image')

    x = inputs
    residuals = []
    for i in range(n_stages):
        n_conv = _get_var(n_conv_per_stage, i)
        if pool_type:
            x = conv_fn(
                x,
                filters     = [_get_var(filters, i)] * n_conv,
                kernel_size = _get_var(kernel_size, i),
                strides     = 1,
                use_bias    = _get_var(use_bias, i),
                padding     = 'same',

                activation  = _get_var(activation, i),

                bnorm       = _get_var(bnorm, i),
                drop_rate   = 0.,

                bn_name = 'down_bn{}'.format(i + 1),
                name    = 'down_conv{}'.format(i + 1)
            )
            residuals.append(x)

            if i < n_stages - 1:
                x = pool_fn(
                    pool_size = _get_var(pool_strides, i), strides = _get_var(pool_strides, i)
                )(x)

            if _get_var(drop_rate, i) > 0:
                x = tf.keras.layers.Dropout(_get_var(drop_rate, i))(x)
            
        else:
            if i < n_stages - 1: n_conv -= 1
            if n_conv <= 0:
                raise ValueError('`n_conv_per_stage` must be > 1 when using strides but got {} at stage {}'.format(
                    n_conv, i
                ))
            x = conv_fn(
                x,
                filters     = [_get_var(filters, i)] * n_conv,
                kernel_size = _get_var(kernel_size, i),
                strides     = 1,
                use_bias    = _get_var(use_bias, i),
                padding     = 'same',

                activation  = _get_var(activation, i),

                bnorm       = _get_var(bnorm, i),
                drop_rate   = 0. if i < n_stages - 1 else _get_var(drop_rate, i),

                bn_name = 'down_bn{}'.format(i + 1),
                name    = 'down_conv{}'.format(i + 1)
            )
            residuals.append(x)

            if i < n_stages - 1:
                x = conv_fn(
                    x,
                    filters     = _get_var(filters, i),
                    kernel_size = _get_var(kernel_size, i),
                    strides     = _get_var(strides, i),
                    use_bias    = _get_var(use_bias, i),
                    padding     = 'same',

                    activation  = _get_var(activation, i),

                    bnorm       = _get_var(bnorm, i),
                    drop_rate   = _get_var(drop_rate, i),

                    bn_name = 'down_bn{}'.format(i + 1),
                    name    = 'down_conv{}_{}'.format(i + 1, n_conv)
                )


    for i in reversed(range(n_stages - 1)):
        x = upsample_fn(
            x,
            filters     = _get_var(filters, i),
            kernel_size = 1 + (_get_var(pool_strides, i) if pool_type else _get_var(strides, i)),
            strides     = _get_var(pool_strides, i) if pool_type else _get_var(strides, i),
            padding     = 'same',
            activation  = _get_var(upsampling_activation, i),
            bnorm       = 'never',
            drop_rate   = 0.,
            name        = 'upsampling_{}'.format(i + 1)
        )
        x = _get_concat_layer(concat_mode)([x, residuals[i]])
        x = conv_fn(
            x,
            filters     = [_get_var(kwargs.get('up_filters', filters), i)] * _get_var(
                kwargs.get('up_n_conv_per_stage', n_conv_per_stage), i
            ),
            kernel_size = _get_var(kwargs.get('up_kernel_size', kernel_size), i),
            use_bias    = _get_var(kwargs.get('up_use_bias', use_bias), i),
            strides     = 1,
            padding     = 'same',
            
            activation  = _get_var(kwargs.get('up_activation', activation), i),

            bnorm       = _get_var(kwargs.get('up_bnorm', bnorm), i),
            drop_rate   = _get_var(kwargs.get('up_drop_rate', drop_rate), i),
            
            bn_name = 'up_bn{}'.format(i + 1),
            name    = 'up_conv{}'.format(i + 1)
        )

    if isinstance(output_dim, (list, tuple)):
        out = [tf.keras.layers.Conv2D(
            filters     = out_dim_i,
            kernel_size = 1,
            strides     = 1,
            activation  = _get_var(final_activation, i),
            name    = '{}_{}'.format(final_name, i + 1) if isinstance(final_name, str) else final_name[i]
        )(x) for i, out_dim_i in enumerate(output_dim)]
    else:
        out = tf.keras.layers.Conv2D(
            output_dim, kernel_size = 1, strides = 1, activation = final_activation, name = final_name
        )(x)
    
    return tf.keras.Model(inputs = inputs, outputs = out, name = name)

custom_functions    = {
    'UNet'    : UNet
}
