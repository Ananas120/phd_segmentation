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

import math
import collections
import numpy as np
import tensorflow as tf

from utils.plot_utils import *

def plot_mask(img, mask, n = 10, ** kwargs):
    if isinstance(mask, tf.sparse.SparseTensor): mask = tf.sparse.to_dense(mask).numpy()
    if hasattr(img, 'numpy'): img = img.numpy()
    if hasattr(mask, 'numpy'): mask = mask.numpy()
    if len(mask.shape) == 4: mask = np.argmax(mask, axis = -1)
    
    indexes = list(range(0, img.shape[-1], max(1, img.shape[-1] // n)))[:n]
    datas = collections.OrderedDict()
    for i, idx in enumerate(indexes):
        frame = img[..., idx]
        
        mask_slice = mask[..., idx : idx + 1]
        mask_alpha = np.repeat(mask_slice, 3, axis = -1)
        mask_alpha = np.concatenate([np.clip(mask_alpha, 0., 1.), 0.4 * (mask_alpha[:, :, :1] != 0)], axis = -1).astype(np.float32)
        
        frame = frame - np.min(frame)
        frame = frame / np.max(frame)
        datas.update({
            'colormap #{}'.format(idx) : {'x' : {'frame' : frame, 'mask' : mask_alpha}},
            'frame #{}'.format(idx)    : frame,
            'mask #{}'.format(idx)     : mask_slice[..., 0].astype(np.int32)
        })
    
    kwargs.setdefault('size', 3.5)
    kwargs.setdefault('ncols', 3)
    kwargs.setdefault('with_colorbar', True)
    
    return plot_multiple(
        ** datas, plot_type = 'imshow', ** kwargs
    )
