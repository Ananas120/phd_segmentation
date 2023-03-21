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
import time
import logging
import functools
import collections
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm as tqdm_progress_bar
from tensorflow.python.keras.callbacks import CallbackList

try:
    import tensorflow_federated as tff
except ImportError:
    def raise_error(* args, ** kwargs):
        raise ImportError('`tensorflow_federated` is not available, : FL features are not available !')
    
    class tff:
        @staticmethod
        def tf_computation(fn):
            return raise_error

        @staticmethod
        def tf_computation(fn):
            return raise_error

from utils import time_to_string
from hparams import HParams, HParamsTraining
from datasets import train_test_split, prepare_dataset
from custom_train_objects import get_callbacks, get_optimizer
from utils.fl_utils import fl_split_client_data, finalize_and_concat_metrics, client_id_to_tf_dataset
from models.interfaces.base_model import BaseModel

logger = logging.getLogger(__name__)

FLTrainingHParams = HParams()

class BaseFLModel(BaseModel):
    def _init_fl(self, ** kwargs):
        self._global_layers_index   = None
        self._local_layers_index    = None

    def split_model_layers(self, model):
        return (
            [model.layers[i] for i in self.global_layers_index],
            [model.layers[i] for i in self.local_layers_index]
        )

    def get_model_fn(self):
        def vanilla_model_fn():
            model   = tf.keras.models.clone_model(server_model)
        
            return tff.learning.from_keras_model(
                model,
                input_spec = (self.input_signature, self.output_signature),
                loss       = loss_fn(),
                metrics    = metrics_fn()
            )
        
        def reconstruction_model_fn():
            model   = tf.keras.models.clone_model(server_model)
        
            global_layers, local_layers = self.split_model_layers(model)

            return tff.learning.reconstruction.from_keras_model(
                model,
                global_layers   = global_layers,
                local_layers    = local_layers,
                input_spec      = (self.input_signature, self.output_signature)
            )
        
        server_model    = self.get_model()
        loss_fn     = self.get_loss_fn()
        metrics_fn  = self.get_metrics_fn()

        if self.use_reconstruction and self._local_layers_index is None:
            layers = server_model.layers
            self._global_layers_index   = [layers.index(l) for l in self.global_layers]
            self._local_layers_index    = [layers.index(l) for l in self.local_layers]

        return (
            functools.partial(reconstruction_model_fn) if self.use_reconstruction else vanilla_model_fn
        )
    
    def get_loss_fn(self):
        loss = self.get_loss()
        return lambda: loss.__class__(** loss.get_config())
    
    def get_metrics_fn(self):
        metrics = self.metrics
        return lambda: [
            metric.__class__(** metric.get_config()) for metric in metrics
        ]
    
    def get_optimizer_fn(self, optimizer_fn = None):
        optimizer = self.get_optimizer()
        if optimizer_fn is None:
            optimizer_fn = lambda: optimizer.__class__(** optimizer.get_config())
        elif not callable(optimizer_fn):
            if isinstance(optimizer_fn, str):
                optimizer_fn = {'optimizer_name' : optimizer_fn}
            elif isinstance(optimizer_fn, (int, float)):
                optimizer_fn = {'lr' : optimizer_fn}
            
            if isinstance(optimizer_fn, dict):
                if 'optimizer_name' not in optimizer_fn:
                    optimizer_fn['optimizer_name'] = optimizer_fn.pop('name', optimizer.__class__.__name__)
                optimizer_fn = get_optimizer(** optimizer_fn)

            optimizer    = optimizer_fn
            optimizer_fn = lambda: optimizer.__class__(** optimizer.get_config())
        return optimizer_fn
    
    def get_server_optimizer_fn(self, optimizer_fn = None):
        return self.get_optimizer_fn(optimizer_fn)
    
    def get_client_optimizer_fn(self, optimizer_fn = None):
        return self.get_optimizer_fn(optimizer_fn)
    
    def get_reconstruction_optimizer_fn(self, optimizer_fn = None):
        return self.get_optimizer_fn(optimizer_fn)
    
    @property
    def use_reconstruction(self):
        return bool(self.local_layers)
    
    @property
    def global_layers(self):
        return None
    
    @property
    def local_layers(self):
        return None
    
    @property
    def global_layers_index(self):
        return self._global_layers_index
    
    @property
    def local_layers_index(self):
        return self._local_layers_index
    
    @property
    def training_hparams_fl(self):
        return FLTrainingHParams()
    
    def _str_fl(self):
        des = ''
        if self.use_reconstruction:
            des += "- Global layers' indexes (n = {}) : {}\n".format(
                len(self.global_layers), self.global_layers_index
            )
            des += "- Local layers' indexes (n = {})  : {}\n".format(
                len(self.local_layers), self.local_layers_index
            )
        return des
    
    def _get_fl_train_config(self,
                             x     = None,
                             y     = None,
                             
                             id_column  = 'id',
                             n_clients  = None,
                             n_train_clients    = None,
                             n_valid_clients    = None,
                             separate_train_and_valid   = False,

                             optimizer_fn   = None,
                             server_optimizer_fn    = None,
                             client_optimizer_fn    = None,
                             reconstruction_optimizer_fn    = None,
                             
                             train_size        = None,
                             valid_size        = None,
                             test_size         = 4,

                             train_times       = 1,
                             valid_times       = 1,

                             pre_shuffle       = False,
                             random_state      = 10,
                             labels            = None,
                             validation_data   = None,

                             test_batch_size   = 1,
                             pred_step         = -1,

                             epochs            = 5, 
                             relative_epoch    = True,

                             verbose           = 1, 
                             callbacks         = [], 
                             ** kwargs
                            ):
        """
            Function that returns dataset configuration for training / evaluation data
            
            Arguments :
                - x / y     : training dataset that will be passed to `prepare_dataset`
                - train_size / valid_size / test_size   : sizes of datasets
                - random_state  : used for shuffling the same way accross trainings
                It allows to pass a complete dataset and the splits will **always** be the same accross trainings which is really interesting to not have problems with shuffling
                - labels        : use in the `sklearn.train_test_split`, rarely used
                - validation_data   : dataset used for validation
                    If not provided, will split training dataset by train_size and valid_size
                    Note that the `test` dataset is a subset of the validation dataset and is used in the `PredictorCallback`
                
                - test_batch_size / pred_step   : kwargs specific for `PredictorCallback`
                
                - epochs    : the number of epochs to train
                - relative_epochs   : whether `epochs` must be seen as absolute epochs (ie the final epoch to reach) or the number of epochs to train on
                
                - verbose   : verbosity level
                - callbacks : training callbacks (added to `self._get_default_callbacks`)
                
                - kwargs    : additional configuration passed to `get_dataset_config` (such as `cache`, `prefetch`, ...)
        """
        if separate_train_and_valid:
            raise NotImplementedError()
        
        if n_train_clients is None: n_train_clients = n_clients
        if n_valid_clients is None: n_valid_clients = n_clients

        if server_optimizer_fn is None: server_optimizer_fn = optimizer_fn
        if client_optimizer_fn is None: client_optimizer_fn = optimizer_fn
        if reconstruction_optimizer_fn is None: reconstruction_optimizer_fn = client_optimizer_fn
            
        train_config = self.get_dataset_config(is_validation = False, ** kwargs)
        valid_config = self.get_dataset_config(is_validation = True, ** kwargs)
        valid_config.setdefault('cache', False)
        
        test_kwargs = kwargs.copy()
        if isinstance(test_batch_size, float):
            test_batch_size = int(valid_config['batch_size'] * test_batch_size)
        test_kwargs.update({
            'batch_size' : test_batch_size, 'train_batch_size' : None, 'valid_batch_size' : None,
            'cache' : False, 'is_validation' : True
        })
        test_config  = self.get_dataset_config(** test_kwargs)
        
        dataset = x if y is None else (x, y)
        if isinstance(dataset, dict) and 'train' in dataset:
            validation_data = dataset.get('valid', dataset.get('test', validation_data))
            dataset         = dataset['train']
        
        if validation_data is None:
            train_dataset, valid_dataset = train_test_split(
                dataset, 
                train_size      = train_size if not separate_train_and_valid else n_train_clients,
                valid_size      = valid_size if not separate_train_and_valid else n_valid_clients,
                random_state    = random_state,
                shuffle         = pre_shuffle,
                labels          = labels,
                split_by_unique = separate_train_and_valid,
                split_column    = id_column
            )
        else:
            train_dataset, _ = train_test_split(
                dataset, train_size = train_size,
                random_state = random_state, shuffle = pre_shuffle
            ) if (train_size) and (not hasattr(dataset, '__len__') or train_size < len(dataset)) else dataset, None
            valid_dataset, _ = train_test_split(
                validation_data, train_size = train_size,
                random_state = random_state, shuffle = pre_shuffle
            ) if (valid_size) and (not hasattr(validation_data, '__len__') or valid_size < len(validation_data)) else validation_data, None
        
        if test_size > 0:
            _, test_dataset = fl_split_client_data(
                valid_dataset, id_column = id_column, n_clients = 1
            )
            test_dataset    = prepare_dataset(test_dataset[0],  ** test_config)
            test_dataset    = test_dataset.take(test_size)
        
        train_ids, train_dataset = fl_split_client_data(
            train_dataset, id_column = id_column, n_clients = n_train_clients
        )
        valid_ids, valid_dataset = fl_split_client_data(
            valid_dataset,
            id_column = id_column,
            n_clients = n_valid_clients if separate_train_and_valid else n_train_clients,
            ids       = train_ids if not separate_train_and_valid else None,
            skip_ids  = train_ids if separate_train_and_valid else None,
        )
        
        if separate_train_and_valid:
            train_idx       = [i for i, c_id in enumerate(train_ids) if c_id in valid_ids]
            train_ids       = [train_ids[idx] for idx in train_idx]
            train_dataset   = [train_dataset[idx] for idx in train_idx]
            
            valid_idx       = [train_ids.index(c_id) for c_id in valid_ids if c_id in train_ids]
            valid_ids       = [valid_ids[idx] for idx in valid_idx]
            valid_dataset   = [valid_dataset[idx] for idx in valid_idx]
            
            if len(train_ids) == 0:
                raise ValueError('`separate_train_and_valid` is True but datasets are disjoint !')
        
        train_dataset = [
            prepare_dataset(client_data, ** train_config) for client_data in train_dataset
        ]
        valid_dataset = [
            prepare_dataset(client_data, ** valid_config) for client_data in valid_dataset
        ]
        
        if train_times > 1: train_dataset = [ds.repeat(train_times) for ds in train_dataset]
        if valid_times > 1: valid_dataset = [ds.repeat(valid_times) for ds in valid_dataset]
        
        if relative_epoch: epochs += self.epochs
        
        callbacks = callbacks + self._get_default_callbacks()
        
        if hasattr(self, 'predict_with_target') and test_size > 0 and pred_step != 0:
            predictor_callback  = get_callbacks(PredictorCallback = {
                'method'    : self.predict_with_target,
                'generator' : test_dataset,
                'directory' : self.train_test_dir,
                'initial_step'  : self.steps,
                'pred_every'    : pred_step
            })
            callbacks += predictor_callback
        
        return {
            'model_fn'  : self.get_model_fn(),
            'loss_fn'   : self.get_loss_fn(),
            'metrics_fn'    : self.get_metrics_fn(),
            'server_optimizer_fn'   : self.get_server_optimizer_fn(server_optimizer_fn),
            'client_optimizer_fn'   : self.get_client_optimizer_fn(client_optimizer_fn),
            'reconstruction_optimizer_fn'   : self.get_reconstruction_optimizer_fn(
                reconstruction_optimizer_fn
            ),

            'train_ids' : train_ids,
            'valid_ids' : valid_ids,
            
            'x'                 : train_dataset,
            'epochs'            : epochs,
            'verbose'           : verbose,
            'callbacks'         : callbacks,
            'validation_data'   : valid_dataset,
            'shuffle'           : False,
            'initial_epoch'     : self.epochs,
            'global_batch_size' : train_config['batch_size']
        }

    @tf.function
    def client_update_step(self, client_model, client_variables, batch, client_optimizer):
        with tf.GradientTape() as tape:
            outputs = client_model.forward_pass(batch, training = True)

        grads = tape.gradient(outputs.loss, client_variables.trainable)
        client_optimizer.apply_gradients(zip(grads, client_variables.trainable))
        
        return outputs.num_examples if outputs.num_examples is not None else tf.shape(output.predictions)[0]
    
    @tf.function
    def client_eval_step(self, client_model, batch):
        outputs = client_model.forward_pass(batch, training = False)
        
        return outputs.num_examples if outputs.num_examples is not None else tf.shape(output.predictions)[0]

    def build_client_update(self, use_experimental_simulation_loop = True):
        def vanilla_client_update(server_weights, client_model, client_data, client_optimizer):
            def get_initial_reduce_state():
                return tf.zeros((), dtype = tf.float32)
            
            @tf.function
            def reduce_fn_train(num_examples_sum, batch):
                num_examples = tf.cast(self.client_update_step(
                    client_model, client_variables, batch, client_optimizer
                ), tf.float32)
                return num_examples_sum + num_examples
            
            @tf.function
            def reduce_fn_valid(num_examples_sum, batch):
                num_examples = tf.cast(self.client_eval_step(
                    client_model, batch
                ), tf.float32)
                return num_examples_sum + num_examples

            
            train_data, valid_data = client_data if isinstance(client_data, tuple) else (client_data, None)

            client_variables = tff.learning.models.ModelWeights.from_model(client_model)

            tf.nest.map_structure(
                lambda w, s_w: w.assign(s_w), client_variables, server_weights
            )

            num_examples = dataset_reduce_fn(
                reduce_fn_train, train_data, initial_state_fn = get_initial_reduce_state
            )

            update = tf.nest.map_structure(
                tf.subtract, server_weights.trainable, client_variables.trainable
            )

            train_metrics = client_model.report_local_unfinalized_metrics()
            
            if valid_data is not None:
                client_model.reset_metrics()
                _ = dataset_reduce_fn(
                    reduce_fn_valid, valid_data, initial_state_fn = get_initial_reduce_state
                )
                valid_metrics = client_model.report_local_unfinalized_metrics()
            else:
                valid_metrics = train_metrics
            
            return tff.learning.templates.ClientResult(
                update = update, update_weight = num_examples
            ), train_metrics, valid_metrics
        
        dataset_reduce_fn = tff.learning.framework.dataset_reduce.build_dataset_reduce_fn(
            use_experimental_simulation_loop
        )

        return tf.function(vanilla_client_update)

    def build_client_work(self,
                          model_fn,
                          optimizer_fn,
                          metrics_aggregator    = None,
                          use_validation_dataset    = False,
                          add_metrics_per_client    = True,
                          use_experimental_simulation_loop  = True
                         ):
        if metrics_aggregator is None: metrics_aggregator = tff.learning.metrics.sum_then_finalize
        
        concat_metrics_fn   = None
        with tf.Graph().as_default():
            model       = model_fn()
            metric_tensors      = model.report_local_unfinalized_metrics()
            metrics_type            = tff.framework.type_from_tensors(metric_tensors)
            metrics_aggregator_fn   = metrics_aggregator(model.metric_finalizers(), metrics_type)

            if add_metrics_per_client:
                metric_tensors['id']    = tf.zeros((), dtype = tf.int32)
                metrics_type_with_id    = tff.framework.type_from_tensors(metric_tensors)
                concat_metrics_fn       = finalize_and_concat_metrics(
                    model.metric_finalizers(), metrics_type_with_id
                )
        
        data_type    = tff.SequenceType(model.input_spec)
        cid_type     = tff.SequenceType(tf.TensorSpec(shape = (), dtype = tf.int32))
        weights_type = tff.learning.models.weights_type_from_model(model)

        if use_validation_dataset: data_type = (data_type, data_type)
        
        client_update   = self.build_client_update(
            use_experimental_simulation_loop = use_experimental_simulation_loop
        )
        
        @tff.federated_computation
        def initialize_fn():
            return tff.federated_value((), tff.SERVER)

        @tff.tf_computation(weights_type, cid_type, data_type)
        def client_update_computation(model_weights, client_id, dataset):
            cid         = client_id.reduce(tf.zeros((), dtype = tf.int32), lambda _, y: y)
            model       = model_fn()
            optimizer   = optimizer_fn()
            return client_update(model_weights, model, dataset, optimizer) + (cid, )

        @tff.tf_computation
        def add_client_id(c_id, metrics):
            metrics['id'] = c_id
            return metrics

        @tff.federated_computation(
            initialize_fn.type_signature.result,
            tff.type_at_clients(weights_type),
            tff.type_at_clients((cid_type, data_type))
        )
        def next_fn(state, model_weights, client_dataset_with_id):
            client_ids, client_dataset = client_dataset_with_id
            client_result, train_metrics, valid_metrics, client_ids = tff.federated_map(
                client_update_computation, (model_weights, client_ids, client_dataset)
            )
            result = collections.OrderedDict(metrics = collections.OrderedDict(
                train = metrics_aggregator_fn(train_metrics)
            ))
            
            if use_validation_dataset:
                result['metrics']['valid'] = metrics_aggregator_fn(valid_metrics)

            if add_metrics_per_client:
                train_metrics_with_id = tff.federated_map(add_client_id, (client_ids, train_metrics))
                result['client_metrics'] = collections.OrderedDict(
                    train = concat_metrics_fn(train_metrics_with_id)
                )
                if use_validation_dataset:
                    valid_metrics_with_id = tff.federated_map(add_client_id, (client_ids, valid_metrics))
                    result['client_metrics']['valid'] = concat_metrics_fn(valid_metrics_with_id)

            measurements = tff.federated_zip(result)
            return tff.templates.MeasuredProcessOutput(state, client_result, measurements)

        return tff.learning.templates.ClientWorkProcess(
            initialize_fn, next_fn
        )
    
    def train_fl(self, 
                 * args,
                 metrics_aggregator = None,
                 use_experimental_simulation_loop = True,
                 
                 verbose       = 1,
                 eval_epoch    = 1,
                 tqdm          = tqdm_progress_bar,
                 ** kwargs
                ):
        ########################################
        #     Initialisation des variables     #
        ########################################
        
        base_hparams    = HParamsTraining().extract(kwargs, pop = False)
        train_hparams   = (self.training_hparams + self.training_hparams_fl).extract(kwargs, pop = True)
        self.init_train_config(** train_hparams)
        
        config = self._get_fl_train_config(* args, ** kwargs)
        self.global_batch_size = config.pop('global_batch_size', -1)

        base_hparams.extract(config, pop = False)
        train_hparams.update(base_hparams)
        
        train_hparams.update({
            'n_clients' : kwargs.get('n_clients', None),
            'n_train_clients' : kwargs.get('n_train_clients', None),
            'n_valid_clients' : kwargs.get('n_valid_clients', None),
            
            'server_optimizer_fn' : kwargs.get('server_optimizer_fn', None),
            'client_optimizer_fn' : kwargs.get('client_optimizer_fn', None),
            'reconstruction_optimizer_fn' : kwargs.get('reconstruction_optimizer_fn', None)
        })

        ##############################
        #     Dataset variables      #
        ##############################
        
        train_dataset = config['x']
        valid_dataset = config['validation_data']
        
        train_ids   = config['train_ids']
        valid_ids   = config['valid_ids']
        
        train_ids_dataset   = client_id_to_tf_dataset(train_ids)
        valid_ids_dataset   = client_id_to_tf_dataset(valid_ids)
        
        assert isinstance(train_dataset, list)
        assert isinstance(valid_dataset, list) or valid_epoch <= 0
        
        init_epoch  = config['initial_epoch']
        last_epoch  = config['epochs']
        
        logger.info("# train clients : {}\n# valid clients : {}\nTraining config :\n{}\n".format(
            len(train_dataset), len(valid_dataset), train_hparams
        ))

        ##############################
        #  Metrics + callbacks init  #
        ##############################
        
        # Prepare callbacks
        callbacks   = config['callbacks']
        callbacks.append(self.history)
        callbacks   = CallbackList(callbacks, add_progbar = verbose > 0, model = self)

        callbacks.set_params(train_hparams)
        
        callbacks.on_train_begin()
        
        ##############################
        #   Iterative process init   #
        ##############################
        
        model    = self.get_model()
        model_fn = config['model_fn']

        pretrained_weights = tff.learning.models.ModelWeights(
            trainable     = [v.numpy() for v in model.trainable_variables],
            non_trainable = [v.numpy() for v in model.non_trainable_variables]
        )
        
        @tff.tf_computation
        def initial_model_weights_fn():
            return pretrained_weights
            #return tff.learning.models.ModelWeights.from_model(model_fn())

        weights_type        = initial_model_weights_fn.type_signature.result
        aggregator_factory  = tff.aggregators.MeanFactory()

        distributor  = tff.learning.templates.build_broadcast_process(weights_type)
        client_work  = self.build_client_work(
            config['model_fn'],
            config['client_optimizer_fn'],
            metrics_aggregator    = metrics_aggregator,
            use_validation_dataset    = valid_dataset is not None,
            add_metrics_per_client    = True,
            use_experimental_simulation_loop  = use_experimental_simulation_loop
        )
        aggregator   = aggregator_factory.create(weights_type.trainable, tff.TensorType(tf.float32))
        finalizer    = tff.learning.templates.build_apply_optimizer_finalizer(
            config['server_optimizer_fn'], weights_type
        )

        training_process = tff.learning.templates.compose_learning_process(
            initial_model_weights_fn,
            distributor,
            client_work,
            aggregator,
            finalizer
        ) 
        
        logger.info(training_process.next.type_signature.formatted_representation())

        ####################
        #     Training     #
        ####################
        
        start_training_time = time.time()
        last_print_time, last_print_step = start_training_time, int(self.current_step.numpy())
        
        try:
            state = training_process.initialize()
            
            for epoch in range(init_epoch, last_epoch):
                logger.info("\nEpoch {} / {}".format(epoch + 1, last_epoch))
                callbacks.on_epoch_begin(epoch)
                
                start_epoch_time = time.time()
                
                sample_dataset = train_dataset if valid_dataset is None else list(zip(train_dataset, valid_dataset))
                
                result = training_process.next(state, list(zip(train_ids_dataset, sample_dataset)))
                state, client_infos = result.state, result.metrics['client_work']
                
                metrics = client_infos['metrics']['train']
                if 'valid' in client_infos['metrics']:
                    metrics.update({
                        'val_{}'.format(name) : value for name, value in client_infos['metrics']['valid'].items()
                    })
                print(metrics)
                
                epoch_time = time.time() - start_epoch_time
                
                self.update_train_config(
                    step    = self.current_step,
                    epoch   = self.current_epoch + 1
                )
                
                callbacks.on_epoch_end(
                    epoch, logs = {
                        name : value for name, value in metrics.items()
                        if 'num_batches' not in name and 'num_examples' not in name
                    }
                )
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted ! Saving model...")
        
        callbacks.on_train_end()
        
        tf.nest.map_structure(
            lambda w, up: w.assign(tf.cast(up, w.dtype)),
            model.trainable_variables,
            state.global_model_weights.trainable
        )
        
        total_training_time = time.time() - start_training_time
        logger.info("Training finished after {} !".format(time_to_string(total_training_time)))
        
        self.save()
        
        return self.history

    def get_config_fl(self, * args, ** kwargs):
        return {}
