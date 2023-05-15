
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

import numpy as np
import tensorflow as tf

from custom_train_objects.losses.dice_loss import dice_coeff

class EASTLoss(tf.keras.losses.Loss):
    def __init__(self,
                 score_factor    = 1.,
                 rbox_factor    = 1.,
                 theta_factor   = 10.,
                 
                 dice_smoothing = 0.01,
                 geo_smoothing  = 1.,
                 from_logits    = True,
                 
                 reduction  = 'none',
                 name   = 'EASTLoss',
                 ** kwargs
                ):
        super().__init__(name = name, reduction = 'none', ** kwargs)
        
        self.from_logits    = from_logits

        self.dice_smoothing = tf.cast(dice_smoothing, tf.float32)
        self.geo_smoothing  = tf.cast(geo_smoothing, tf.float32)
        
        self.score_factor   = tf.cast(score_factor, tf.float32)
        self.rbox_factor    = tf.cast(rbox_factor, tf.float32)
        self.theta_factor   = tf.cast(theta_factor, tf.float32)
    
        self.pi = tf.cast(np.pi, tf.float32)
    
    @property
    def metric_names(self):
        return ['loss', 'score_loss', 'geo_loss', 'theta_loss']
    
    def score_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        return 1. - dice_coeff(
            tf.reshape(y_true, [batch_size, -1, 1]),
            tf.reshape(y_pred, [batch_size, -1, 1]),
            self.dice_smoothing
        )
    
    def geo_loss(self, y_true, y_pred, mask):
        pred_d0, pred_d1, pred_d2, pred_d3 = (
            y_pred[..., 0], y_pred[..., 1], y_pred[..., 2], y_pred[..., 3]
        )

        if self.from_logits:
            height  = tf.cast(tf.shape(y_true)[1], tf.float32)
            width   = tf.cast(tf.shape(y_true)[2], tf.float32)
            
            pred_d0 = pred_d0 * height
            pred_d1 = pred_d1 * width
            pred_d2 = pred_d2 * height
            pred_d3 = pred_d3 * width
        
        true_area   = (y_true[..., 0] + y_true[..., 2]) * (y_true[..., 1] + y_true[..., 3])
        pred_area   = (pred_d0 + pred_d2) * (pred_d1 + pred_d3)

        h_union = tf.minimum(y_true[..., 0], pred_d0) + tf.minimum(y_true[..., 2], pred_d2)
        w_union = tf.minimum(y_true[..., 1], pred_d1) + tf.minimum(y_true[..., 3], pred_d3)
        
        intersect   = w_union * h_union
        union       = true_area + pred_area - intersect

        if tf.reduce_any(intersect < 0) or tf.reduce_any(union < 0):
            tf.print(tf.reduce_min(intersect), tf.reduce_min(union))
            #raise ValueError('negative value {} - {} !'.format(
            #    tf.reduce_min(intersect), tf.reduce_min(union)
            #))
        
        geo_loss    = - tf.math.log(tf.maximum(
            1e-6, (intersect + self.geo_smoothing) / (union + self.geo_smoothing)
        ))
        
        return geo_loss * mask
    
    def theta_loss(self, true_theta_map, pred_theta_map, mask):
        if self.from_logits:
            pred_theta_map  = (pred_theta_map - 0.5) / self.pi * 2.

        theta_loss = 1. - tf.math.cos(true_theta_map - pred_theta_map)
        
        return theta_loss * mask
    
    def call(self, y_true, y_pred):
        true_score_map, true_geo_map, valid_mask = y_true
        pred_score_map, pred_geo_map = y_pred
        
        if len(tf.shape(true_score_map)) == 3:
            true_score_map = tf.expand_dims(true_score_map, axis = -1)
        
        batch_size  = tf.shape(true_score_map)[0]
        
        true_score_map  = tf.where(tf.expand_dims(valid_mask, axis = -1), true_score_map, 0.)
        pred_score_map  = tf.where(tf.expand_dims(valid_mask, axis = -1), pred_score_map, 0.)

        n_valid     = tf.maximum(tf.reduce_sum(tf.reshape(
            true_score_map, [batch_size, -1]
        ), axis = -1), 1.)

        score_loss  = self.score_loss(true_score_map, pred_score_map)
        geo_loss    = self.geo_loss(
            true_geo_map[..., :4], pred_geo_map[..., :4], true_score_map[..., 0]
        )
        theta_loss  = self.theta_loss(
            true_geo_map[..., 4], pred_geo_map[..., 4], true_score_map[..., 0]
        )
        
        score_loss  = score_loss * self.score_factor
        geo_loss    = tf.reduce_sum(
            tf.reshape(geo_loss, [batch_size, -1]), axis = -1
        ) * self.rbox_factor / n_valid
        theta_loss  = tf.reduce_sum(
            tf.reshape(theta_loss, [batch_size, - 1]), axis = -1
        ) * self.theta_factor / n_valid
        
        loss    = score_loss + geo_loss + theta_loss
        
        return loss, score_loss, geo_loss, theta_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'from_logits'    : self.from_logits,

            'dice_smoothing' : self.dice_smoothing.numpy(),
            'geo_smoothing'  : self.geo_smoothing.numpy(),
            
            'score_factor'   : self.score_factor.numpy(),
            'rbox_factor'    : self.rbox_factor.numpy(),
            'theta_factor'   : self.theta_factor.numpy()
        })
        return config