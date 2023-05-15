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

from utils import compute_centroids
from utils.distance import distance

class GE2ESegLoss(tf.keras.losses.Loss):
    def __init__(self,
                 mode = 'softmax',
                 distance_metric    = 'cosine',
                 
                 background_mode = 'ignore',
                 
                 init_w = 1.,
                 init_b = 0.,

                 name = 'ge2e_seg_loss',
                 
                 ** kwargs
                ):
        assert mode in ('softmax', 'contrast')
        
        super().__init__(name = name, ** kwargs)
        self.mode   = mode
        self.distance_metric    = distance_metric
        self.background_mode    = background_mode
        
        if mode == 'softmax':
            self.loss_fn = self.softmax_loss
        else:
            self.loss_fn = self.contrast_loss
        
        self.w = tf.Variable(init_w, trainable = True, dtype = tf.float32, name = 'weight')
        self.b = tf.Variable(init_b, trainable = True, dtype = tf.float32, name = 'bias')
    
    @property
    def variables(self):
        return [self.w, self.b]
    
    @property
    def trainable_variables(self):
        return [self.w, self.b]

    @property
    def metric_names(self):
        return ['loss', 'foreground_loss', 'background_loss']
    
    def compute_centroids(self, mask, embeddings):
        embeddings = tf.gather_nd(embeddings, mask.indices[:, :-1])
        _, idx     = tf.unique(mask.indices[:, -1])
        idx        = tf.cast(idx, tf.int32)

        centroids_ids, centroids  = compute_centroids(embeddings, idx)

        return embeddings, centroids, idx
        
    def similarity_matrix(self, embeddings, centroids):
        return distance(
            embeddings,
            centroids,
            method  = self.distance_metric,
            force_distance  = False,
            max_matrix_size = -1,
            as_matrix       = True
        )
    
    def softmax_loss(self, idx, similarity_matrix):
        return tf.keras.losses.sparse_categorical_crossentropy(
            idx, similarity_matrix, from_logits = True
        )
    
    def contrast_loss(self, idx, similarity_matrix):
        target_matrix = tf.one_hot(idx, depth = tf.shape(similarity_matrix)[-1])
        return tf.reduce_mean(tf.reshape(tf.keras.losses.binary_crossentropy(
            tf.reshape(target_matrix, [-1, 1]), tf.sigmoid(tf.reshape(similarity_matrix, [-1, 1]))
        ), [-1, tf.shape(similarity_matrix)[-1]]), axis = -1)
    
    def foreground_loss(self, y_true, y_pred, embeddings, centroids, idx):
        similarity_matrix = self.similarity_matrix(embeddings, centroids)
        similarity_matrix = similarity_matrix * self.w + self.b
        # shape = (n_embeddings, n_labels)
        # labels = (n_embeddings, )
        loss = self.loss_fn(idx, similarity_matrix)
        
        if tf.shape(y_pred)[0] > 1:
            loss = tf.squeeze(compute_centroids(
                tf.expand_dims(loss, axis = 1), tf.cast(y_true.indices[:, 0], tf.int32)
            )[1], axis = 1)
        else:
            loss = tf.reduce_mean(loss, axis = -1, keepdims = True)
        
        return loss

    def background_loss(self, y_true, y_pred, centroids):
        mask = tf.reduce_any(tf.sparse.to_dense(tf.cast(y_true, tf.bool)), axis = -1, keepdims = True)
        mask = tf.logical_not(mask)
        
        embeddings = tf.boolean_mask(
            tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]]), tf.reshape(mask, [-1])
        )
        
        if self.background_mode == 'clusterize':
            centroids = tf.concat([tf.reduce_mean(embeddings, axis = 0, keepdims = True), centroids], axis = 0)
        
        matrix     = self.similarity_matrix(
            embeddings, centroids
        ) * self.w + self.b
        
        if self.background_mode == 'ignore':
            loss = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(matrix), matrix, from_logits = True
            )
        elif self.background_mode == 'clusterize':
            loss = self.softmax_loss(
                tf.zeros((tf.shape(matrix)[0], ), dtype = tf.int32), matrix
            )
        
        if tf.shape(y_pred)[0] > 1:
            loss = tf.squeeze(compute_centroids(
                tf.expand_dims(loss, axis = 1), tf.cast(y_true.indices[:, 0], tf.int32)
            )[1], axis = 1)
        else:
            loss = tf.reduce_mean(loss, axis = -1, keepdims = True)
        
        return loss
    
    def call(self, y_true, y_pred):
        embeddings, centroids, idx = self.compute_centroids(y_true, y_pred)
        
        if tf.reduce_any(tf.shape(embeddings) == 0):
            tf.print('Embeddings shape :', tf.shape(embeddings), '- centroids shape :', tf.shape(centroids), '- indices shape :', tf.shape(y_true.indices))
            loss = tf.zeros((tf.shape(y_pred)[0], ), dtype = tf.float32)
            return loss, loss, loss

        centroids = tf.stop_gradient(centroids)
        
        fore_loss = self.foreground_loss(y_true, y_pred, embeddings, centroids, idx)
        back_loss = self.background_loss(y_true, y_pred, centroids)

        return fore_loss + back_loss, fore_loss, back_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'mode'  : self.mode,
            'init_w'    : self.w.value().numpy(),
            'init_b'    : self.b.value().numpy(),
            'distance_metric'   : self.distance_metric,
            'background_mode'   : self.background_mode
        })
        return config