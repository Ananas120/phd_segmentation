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
import glob

def __load():
    for module_name in glob.glob(os.path.join('datasets', 'custom_datasets', 'medical_datasets', '*.py')):
        if module_name.endswith('__init__.py'): continue
        module_name = module_name.split('.')[0].replace(os.path.sep, '.')

        module = __import__(module_name)

__load()