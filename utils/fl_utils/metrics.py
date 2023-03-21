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

import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

def finalize_and_concat_metrics(finalizers, metrics_type):
    compiled_finalizers = collections.OrderedDict()
    for k, fn in finalizers.items():
        compiled_finalizers[k] = tff.tf_computation(
            fn, metrics_type[k]
        )
    compiled_finalizers['id'] = tff.tf_computation(lambda x: x, metrics_type['id'])
    
    finalized_spec = tf.nest.map_structure(
        lambda fn: tf.TensorSpec(shape = (None,), dtype = fn.type_signature.result.dtype), compiled_finalizers
    )

    @tff.tf_computation(metrics_type)
    def finalize(unfinalized_metrics):
        finalized = collections.OrderedDict()
        for name, value in unfinalized_metrics.items():
            finalized[name] = finalizers[name](value) if name in finalizers else value
        return finalized
    
    @tff.tf_computation(finalized_spec, metrics_type)
    def accumulate(accumulator, values):
        return tf.nest.map_structure(
            lambda x, y: tf.concat([x, tf.expand_dims(y, 0)], 0), accumulator, finalize(values)
        )

    @tff.tf_computation
    def get_init_state():
        return tf.nest.map_structure(
            lambda m: tf.zeros((1,), dtype = m.dtype), finalized_spec
        )

    @tff.federated_computation(tff.type_at_clients(metrics_type))
    def finalize_and_concat(metrics):
        return tff.federated_aggregate(
            metrics,
            get_init_state(),
            accumulate = accumulate,
            merge      = tff.tf_computation(lambda x, y: y),
            report     = tff.tf_computation(lambda x: tf.nest.map_structure(lambda v: v[1:], x))
        )
    
    return finalize_and_concat
