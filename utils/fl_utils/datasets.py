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

def fl_split_client_data(dataset, id_column = 'id', n_clients = None, ids = None,
                         skip_ids = None, random_state = 10):
    def select_clients(clients):
        if skip_ids is not None: clients = [c for c in clients if c not in skip_ids]
        if ids is not None: clients = [c for c in clients if c in ids]
        
        if n_clients and n_clients > 0 and n_clients < len(clients):
            rnd = np.random.RandomState(random_state)
            clients = rnd.choice(clients, size = n_clients)
        
        return clients
        
    if isinstance(dataset, pd.DataFrame):
        assert id_column in dataset.columns, '{} not in {}'.format(id_column, dataset.columns)
        
        clients = select_clients(dataset[id_column].unique())
        
        return list(clients), [
            dataset[dataset[id_column] == client_id] for client_id in clients
        ]
    elif isinstance(dataset, tff.simulation.datasets.ClientData):
        clients = select_clients(dataset.client_ids)
        
        return list(clients), [
            dataset.create_tf_dataset_for_client(client_id) for client_id in clients
        ]

def client_id_to_tf_dataset(client_ids):
    return [
        tf.data.Dataset.from_tensor_slices(np.array([client_ids.index(id_i)], dtype = np.int32))
        for id_i in client_ids
    ]
