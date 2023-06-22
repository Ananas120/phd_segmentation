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

import numpy as np
import tensorflow as tf

def sparse_argmax(sp, dtype = tf.int32):
    return tf.tensor_scatter_nd_update(
        tf.zeros(sp.dense_shape[:-1], dtype = dtype), sp.indices[:, :-1], tf.cast(sp.indices[:, -1], dtype)
    )

def sparse_pad(sp, padding):
    padding = tf.cast(padding, sp.indices.dtype)
    
    if tf.shape(sp.indices)[1] > tf.shape(padding)[0]:
        padding = tf.concat([
            padding, tf.zeros((tf.shape(sp.indices)[1] - tf.shape(padding)[0], 2), dtype = padding.dtype)
        ], axis = 0)
    
    new_indices = sp.indices + tf.expand_dims(padding[:, 0], axis = 0)
    new_shape   = tf.shape(sp)
    new_shape   = new_shape + tf.cast(tf.reduce_sum(padding, -1), new_shape.dtype)
                
    return tf.sparse.SparseTensor(
        indices = new_indices, values = sp.values, dense_shape = tf.cast(new_shape, tf.int64)
    )
