
# Copyright (C) 2023 Langlois Quentin, ICTEAM, UCLouvain. All rights reserved.
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

from custom_train_objects.losses.ge2e_seg_loss import average_loss

@tf.function(input_signature = [
    tf.SparseTensorSpec(shape = (None, None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (None, None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.bool)
])
def sparse_dice_coeff(y_true, y_pred, smoothing = 0.01, threshold = 0., skip_empty = False):
    """
        Computes the Dice coefficient for possibly multi-label segmentation
        
        Arguments :
            - y_true    : 3-D Tensor ([batch_size, -1, n_labels]), the true segmentation
            - y_pred    : 3-D Tensor ([batch_size, -1, n_labels]), the predicted segmentation
            - smoothing : smoothing value
            - skip_empty    : whether to skip where `y_true == 0` in the whole segmentation for a given label
                /!\ WARNING : if the segmentation is empty for all the classes, the result will be 0
        Return :
            - dice_coeff    : score between [0 (bad), 1 (perfect)]
    """
    # shape = [batch_size, n_labels]
    if threshold > 0.:
        y_pred = tf.where(y_pred < threshold, 0., y_pred)
        y_pred = tf.where(y_pred > 1. - threshold, 1., y_pred)
    
    intersect   = tf.sparse.reduce_sum(y_true * y_pred, axis = 1)
    union       = tf.sparse.reduce_sum(y_true, axis = 1) + tf.reduce_sum(y_pred, axis = 1)

    dice_coeff  = (2. * intersect + smoothing) / tf.maximum(union + smoothing, 1.)
    
    # shape = [batch_size]
    if skip_empty:
        non_empty   = tf.cast(tf.sparse.reduce_sum(y_true, axis = 1) > 0, tf.float32)
        
        dice_coeff  = tf.reduce_sum(dice_coeff * non_empty, axis = -1) / tf.maximum(
            1., tf.reduce_sum(non_empty, axis = -1)
        )
    else:
        dice_coeff = tf.reduce_mean(dice_coeff, axis = -1)
    
    return dice_coeff

@tf.function(input_signature = [
    tf.TensorSpec(shape = (None, None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (None, None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.bool)
])
def dice_coeff(y_true, y_pred, smoothing = 0.01, threshold = 0., skip_empty = False):
    """
        Computes the Dice coefficient for possibly multi-label segmentation
        
        Arguments :
            - y_true    : 3-D Tensor ([batch_size, -1, n_labels]), the true segmentation
            - y_pred    : 3-D Tensor ([batch_size, -1, n_labels]), the predicted segmentation
            - smoothing : smoothing value
            - skip_empty    : whether to skip where `y_true == 0` in the whole segmentation
                /!\ WARNING : if the segmentation is empty for all the classes, the result will be 0
        Return :
            - dice_coeff    : score between [0 (bad), 1 (perfect)]
    """
    if threshold > 0.:
        y_pred = tf.where(y_pred < threshold, 0., y_pred)
        y_pred = tf.where(y_pred > 1. - threshold, 1., y_pred)

    # shape = [batch_size, n_labels]
    intersect   = tf.reduce_sum(y_true * y_pred, axis = 1)
    union       = tf.reduce_sum(y_true, axis = 1) + tf.reduce_sum(y_pred, axis = 1)

    dice_coeff  = (2. * intersect + smoothing) / (union + smoothing)
    
    # shape = [batch_size]
    if skip_empty:
        non_empty   = tf.reduce_sum(y_true, axis = 1) > 0
        
        dice_coeff  = tf.reduce_sum(tf.where(non_empty, dice_coeff, 0.), axis = -1)
        dice_coeff  = dice_coeff / tf.maximum(
            1., tf.reduce_sum(tf.cast(non_empty, tf.float32), axis = -1)
        )
    else:
        dice_coeff = tf.reduce_mean(dice_coeff, axis = -1)
    
    return dice_coeff

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self,
                 smoothing  = 0.01,
                 threshold  = 0.,
                 from_logits    = False,
                 skip_empty_frames  = False,
                 skip_empty_labels  = False,
                 
                 reduction  = 'none',
                 name   = 'DiceLoss',
                 ** kwargs
                ):
        super().__init__(name = name, reduction = 'none', ** kwargs)
        self.smoothing  = tf.cast(smoothing, tf.float32)
        self.threshold  = tf.cast(threshold, tf.float32)
        self.from_logits    = from_logits
        
        self.skip_empty_labels = tf.Variable(
            skip_empty_labels, trainable = False, dtype = tf.bool, name = 'skip_empty_labels'
        )
        self.skip_empty_frames = tf.Variable(
            skip_empty_frames, trainable = False, dtype = tf.bool, name = 'skip_empty_frames'
        )
    
    @property
    def metric_names(self):
        return ['loss', 'dice_coeff']
    
    def call(self, y_true, y_pred, skip_empty_frames = None, skip_empty_labels = None):
        if skip_empty_frames is None: skip_empty_frames = self.skip_empty_frames
        if skip_empty_labels is None: skip_empty_labels = self.skip_empty_labels
        
        is_sparse   = isinstance(y_true, tf.sparse.SparseTensor)
        batch_size, n_classes = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if self.from_logits: y_pred = tf.nn.sigmoid(y_pred)

        if skip_empty_frames and len(tf.shape(y_pred)) == 5:
            if is_sparse:
                y_true = tf.sparse.reshape(y_true, [batch_size, -1, tf.shape(y_pred)[-2], n_classes])
                valid_frames = tf.sparse.reduce_sum(y_true, axis = 1, keepdims = True) > 0
            else:
                y_true = tf.reshape(y_true, [batch_size, -1, tf.shape(y_pred)[-2], n_classes])
                valid_frames = tf.reduce_sum(y_true, axis = 1, keepdims = True) > 0
            
            # shape (after expand_dims) == [batch_size, 1, 1, n_frames, n_classes]
            # values equal 1 iff at least one voxel in this batch, in this frame and in this label is non-zero
            valid_frames.set_shape([None, None, None, None])
            valid_frames = tf.expand_dims(valid_frames, axis = 2)
            y_pred = y_pred * tf.cast(valid_frames, y_pred.dtype)

        if is_sparse:
            y_true = tf.sparse.reshape(y_true, [batch_size, -1, n_classes])
            dice_coeff_fn = sparse_dice_coeff
        else:
            y_true = tf.reshape(y_true, [batch_size, -1, n_classes])
            dice_coeff_fn = dice_coeff
        
        dice = dice_coeff_fn(
            y_true,
            tf.reshape(y_pred, [batch_size, -1, n_classes]),
            smoothing  = self.smoothing,
            threshold  = self.threshold,
            skip_empty = skip_empty_labels or skip_empty_frames
        )
        dice.set_shape([None])
        return 1. - dice, dice
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'smoothing'   : self.smoothing,
            'threshold'   : self.threshold,
            'from_logits' : self.from_logits
        })
        return config
    
class SparseDiceLoss(DiceLoss):
    def call(self, y_true, y_pred):
        n_labels    = tf.shape(y_pred)[-1]
        n_labels    = tf.reshape(tf.range(n_labels), [1, 1, 1, n_labels])
        
        y_true  = tf.cast(tf.expand_dims(y_true, axis = -1) == n_labels, tf.float32)
        return super().call(y_true, y_pred)

class DiceWithBCELoss(DiceLoss):
    def __init__(self, dice_weight = 0.5, bce_weight = 0.5, ** kwargs):
        super().__init__(** kwargs)
        
        self.dice_weight = tf.Variable(dice_weight, trainable = False, dtype = tf.float32, name = 'dice_weight')
        self.bce_weight  = tf.Variable(bce_weight,  trainable = False, dtype = tf.float32, name = 'bce_weight')
    
    @property
    def metric_names(self):
        return ['loss', 'dice', 'bce']
    
    def call(self, y_true, y_pred, ** kwargs):
        dice_loss = tf.zeros((tf.shape(y_pred)[0], ), dtype = tf.float32)
        bce_loss  = tf.zeros((tf.shape(y_pred)[0], ), dtype = tf.float32)
        
        batch_size = tf.shape(y_pred)[0]
        
        y_true = tf.cast(y_true, tf.float32)
        if self.dice_weight > 0.:
            dice_loss += super().call(y_true, y_pred, ** kwargs)[0] * self.dice_weight
        
        if self.bce_weight > 0.:
            if isinstance(y_true, tf.sparse.SparseTensor): y_true = tf.sparse.to_dense(y_true)
            bce_loss += tf.keras.losses.binary_crossentropy(
                tf.reshape(y_true, [batch_size, -1]), tf.reshape(y_pred, [batch_size, -1]), from_logits = self.from_logits
            ) * self.bce_weight
        
        return dice_loss + bce_loss, dice_loss, bce_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'dice_weight' : self.dice_weight.value(),
            'bce_weight'  : self.bce_weight.value()
        })
        return config
    
class DiceWithCELoss(DiceLoss):
    def __init__(self,
                 dice_weight    = 0.5,
                 ce_weight      = 0.5,
                 loss_averaging = 'macro',
                 
                 bce_weight     = None,
                 
                 ** kwargs
                ):
        super().__init__(** kwargs)
        self.loss_averaging = loss_averaging
        
        if bce_weight is not None: ce_weight = bce_weight
        
        self.dice_weight = tf.Variable(
            dice_weight, trainable = False, dtype = tf.float32, name = 'dice_weight'
        )
        self.ce_weight   = tf.Variable(
            ce_weight,  trainable = False, dtype = tf.float32, name = 'ce_weight'
        )
    
    @property
    def metric_names(self):
        return ['loss', 'dice_loss', 'cross_entropy_loss']
    
    def call(self, y_true, y_pred, ** kwargs):
        batch_size  = tf.shape(y_pred)[0]
        dice_loss   = tf.zeros((batch_size, ), dtype = tf.float32)
        ce_loss     = tf.zeros((batch_size, ), dtype = tf.float32)
        
        y_true = tf.cast(y_true, tf.float32)
        if self.dice_weight > 0.:
            dice_loss = super().call(y_true, y_pred, ** kwargs)[0] * self.dice_weight
        
        if self.ce_weight > 0.:
            if isinstance(y_true, tf.sparse.SparseTensor):
                y_true = tf.tensor_scatter_nd_update(
                    tf.zeros(y_true.dense_shape[:-1], dtype = tf.int32),
                    y_true.indices[:, :-1],
                    tf.cast(y_true.indices[:, -1], tf.int32)
                )
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
                tf.reshape(y_true, [batch_size, -1]),
                tf.reshape(y_pred, [batch_size, -1, tf.shape(y_pred)[-1]]),
                from_logits = self.from_logits
            )
            
            if batch_size == 1:
                batch_indexes = tf.zeros((tf.shape(ce_loss)[1], ), dtype = tf.int32)
            else:
                batch_indexes = tf.repeat(
                    tf.range(batch_size, dtype = tf.int32), tf.shape(ce_loss)[1]
                )
            
            ce_loss = average_loss(
                tf.reshape(ce_loss, [-1]),
                ids = tf.reshape(y_true, [-1]),
                mode    = self.loss_averaging,
                batch_size  = batch_size,
                batch_idx   = batch_indexes
            ) * self.ce_weight
        
        return dice_loss + ce_loss, dice_loss, ce_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'dice_weight' : self.dice_weight.value(),
            'bce_weight'  : self.ce_weight.value(),
            'loss_averaging'    : self.loss_averaging
        })
        return config