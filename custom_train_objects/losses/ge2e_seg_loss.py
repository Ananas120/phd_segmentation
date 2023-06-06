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

def average_loss(losses, batch_size, batch_idx, ids, mode = 'macro'):
    if batch_size == 1:
        if mode == 'macro':
            return tf.reduce_mean(losses, axis = -1, keepdims = True)
        elif mode == 'micro':
            return tf.reduce_mean(tf.squeeze(compute_centroids(
                tf.expand_dims(losses, axis = 1), tf.cast(ids, tf.int32)
            )[1], axis = -1), axis = -1, keepdims = True)
    else:
        if mode == 'macro':
            return tf.squeeze(compute_centroids(
                tf.expand_dims(losses, axis = 1), tf.cast(batch_idx, tf.int32)
            )[1], axis = -1)
        else:
            n_labels    = tf.reduce_max(ids)
            indexes     = tf.cast(batch_idx * n_labels + ids, tf.int32)
            loss_label_batch_idx, loss_per_label_per_batch = compute_centroids(
                tf.expand_dims(losses, axis = 1), indexes
            )

            loss_batch_idx = loss_label_batch_idx // n_labels
            return tf.squeeze(compute_centroids(
                loss_per_label_per_batch, loss_batch_idx
            )[1], axis = 1)

class GE2ESegLoss(tf.keras.losses.Loss):
    def __init__(self,
                 mode   = 'softmax',
                 distance_metric    = 'cosine',
                 
                 background_mode    = 'ignore',
                 
                 loss_averaging = 'macro',
                 
                 init_w = 1.,
                 init_b = 0.,

                 name = 'ge2e_seg_loss',
                 
                 ** kwargs
                ):
        assert mode in ('softmax', 'contrast')
        assert loss_averaging in ('micro', 'macro')
        assert background_mode in ('concat', 'clusterize', 'ignore')
        
        super().__init__(name = name, ** kwargs)
        self.mode   = mode
        self.loss_averaging = loss_averaging
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
    
    def similarity_matrix(self, embeddings, centroids):
        return distance(
            embeddings,
            centroids,
            method  = self.distance_metric,
            force_distance  = False,
            max_matrix_size = 1024 ** 2 * 64,
            as_matrix       = True
        )

    def softmax_loss(self, idx, similarity_matrix):
        return tf.keras.losses.sparse_categorical_crossentropy(
            idx, similarity_matrix, from_logits = True
        )
    
    def contrast_loss(self, idx, similarity_matrix):
        target_matrix = tf.one_hot(idx, depth = tf.shape(similarity_matrix)[-1])
        return tf.keras.losses.binary_crossentropy(
            target_matrix, similarity_matrix, from_logits = True
        )

    def compute_foreground_centroids(self, mask, embeddings):
        embeddings  = tf.gather_nd(embeddings, mask.indices[:, :-1])
        indexes     = tf.cast(tf.unique(mask.indices[:, -1])[1], tf.int32)

        centroids_ids, centroids  = compute_centroids(embeddings, indexes)

        return embeddings, centroids, indexes
    
    def compute_background_centroid(self, mask, embeddings):
        mask = tf.sparse.reduce_sum(mask, axis = -1, keepdims = True) == 0
        
        embeddings  = tf.boolean_mask(
            tf.reshape(embeddings, [-1, tf.shape(embeddings)[-1]]), tf.reshape(mask, [-1])
        )
        centroid    = tf.reduce_mean(embeddings, axis = 0, keepdims = True)
        return embeddings, centroid, mask
    
    def foreground_loss(self,
                        y_true,
                        y_pred,
                        fore_ids,
                        fore_embeddings,
                        back_embeddings,
                        fore_centroids,
                        back_centroids,
                        all_centroids
                       ):
        """
            Computes the foreground part of the loss
            
            Arguments :
                - y_true    : `tf.SparseTensor` containing the indices of the foreground items
                - y_pred    : `tf.Tensor` whose last dimension is `embedding_dim`
                - fore_idx  : `tf.Tensor` with shape [n_fore_embeddings], the id associated to each `fore_embeddings`
                - fore_embeddings   : `tf.Tensor` with shape [n_fore_embeddings, embedding_dim], the embeddings of foreground pixels / voxels
                - back_embeddings   : `tf.Tensor` with shape [n_back_embeddings, embedding_dim]
                - fore_centroids    : `tf.Tensor` with shape [len(np.unique(fore_idx)), embedding_dim], the centroid for each id in `fore_idx`
                - back_centroids    : `tf.Tensor` with shape [1, embedding_dim], the centroid of all the background embeddings
                - all_centroids     : concatenation of `[back_centroids, fore_centroids]`
            Return :
                - fore_loss : `tf.Tensor` with shape [batch_size]
        """
        if self.background_mode == 'concat':
            centroids   = all_centroids
            fore_ids    = fore_ids + 1
        else:
            centroids   = fore_centroids
        
        # shape = [n_fore_embeddings, len(centroids)]
        similarity_matrix = self.similarity_matrix(fore_embeddings, centroids)
        similarity_matrix = similarity_matrix * self.w + self.b
        # shape = (n_embeddings, n_labels)
        # labels = (n_embeddings, )
        loss_per_embedding = self.loss_fn(fore_ids, similarity_matrix)
        
        return average_loss(
            loss_per_embedding,
            mode    = self.loss_averaging,
            ids     = fore_ids,
            batch_size  = tf.shape(y_pred)[0],
            batch_idx   = tf.cast(y_true.indices[:, 0], tf.int32)
        )

    def background_loss(self,
                        y_true,
                        y_pred,
                        fore_ids,
                        fore_embeddings,
                        back_embeddings,
                        fore_centroids,
                        back_centroids,
                        all_centroids,
                        back_mask
                       ):
        """
            Computes the foreground part of the loss
            
            Arguments :
                - y_true    : `tf.SparseTensor` containing the indices of the foreground items
                - y_pred    : `tf.Tensor` whose last dimension is `embedding_dim`
                - fore_idx  : `tf.Tensor` with shape [n_fore_embeddings], the id associated to each `fore_embeddings`
                - fore_embeddings   : `tf.Tensor` with shape [n_fore_embeddings, embedding_dim], the embeddings of foreground pixels / voxels
                - back_embeddings   : `tf.Tensor` with shape [n_back_embeddings, embedding_dim]
                - fore_centroids    : `tf.Tensor` with shape [len(np.unique(fore_idx)), embedding_dim], the centroid for each id in `fore_idx`
                - back_centroids    : `tf.Tensor` with shape [1, embedding_dim], the centroid of all the background embeddings
                - all_centroids     : concatenation of `[back_centroids, fore_centroids]`
            Return :
                - fore_loss : `tf.Tensor` with shape [batch_size]
        """
        centroids = fore_centroids
        if self.background_mode != 'ignore':
            centroids = all_centroids
        
        matrix     = self.similarity_matrix(
            back_embeddings, centroids
        ) * self.w + self.b
        
        if self.background_mode == 'ignore':
            loss_per_embedding = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(matrix), matrix, from_logits = True
            )
        else:
            loss_per_embedding = self.softmax_loss(
                tf.zeros((tf.shape(matrix)[0], ), dtype = tf.int32), matrix
            )
        
        if tf.shape(y_pred)[0] == 1:
            batch_indexes = tf.zeros((tf.shape(back_embeddings)[0], ), dtype = tf.int32)
        else:
            batch_indexes = tf.cast(tf.where(back_mask)[:, 0], tf.int32)
        
        return average_loss(
            loss_per_embedding,
            ids     = None,
            mode    = 'macro',
            batch_size  = tf.shape(y_pred)[0],
            batch_idx   = batch_indexes
        )
    
    def call(self, y_true, y_pred):
        fore_embeddings, fore_centroids, fore_idx = self.compute_foreground_centroids(y_true, y_pred)

        if tf.reduce_any(tf.shape(fore_embeddings) == 0):
            tf.print('Embeddings shape :', tf.shape(fore_embeddings), '- centroids shape :', tf.shape(fore_centroids), '- indices shape :', tf.shape(y_true.indices))
            loss = tf.zeros((tf.shape(y_pred)[0], ), dtype = tf.float32)
            return loss, loss, loss
        
        back_embeddings, back_centroids, back_mask = self.compute_background_centroid(y_true, y_pred)

        centroids   = tf.concat([back_centroids, fore_centroids], axis = 0)
        
        fore_loss = self.foreground_loss(
            y_true,
            y_pred,
            fore_idx,
            fore_embeddings,
            back_embeddings,
            fore_centroids,
            back_centroids,
            centroids
        )
        back_loss = self.background_loss(
            y_true,
            y_pred,
            fore_idx,
            fore_embeddings,
            back_embeddings,
            fore_centroids,
            back_centroids,
            centroids,
            back_mask
        )

        return fore_loss + back_loss, fore_loss, back_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'mode'  : self.mode,
            'init_w'    : self.w.value().numpy(),
            'init_b'    : self.b.value().numpy(),
            'loss_averaging'    : self.loss_averaging,
            'distance_metric'   : self.distance_metric,
            'background_mode'   : self.background_mode
        })
        return config