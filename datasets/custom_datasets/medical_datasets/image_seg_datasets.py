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
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from utils.file_utils import load_json, dump_json
from datasets.custom_datasets import get_dataset_dir

logger = logging.getLogger(__name__)

TCIA_DIR = '{}/TCIA'.format(get_dataset_dir())

IMAGE_SEGMENTATION = 'Organ segmentation'

def image_seg_dataset_wrapper(name, task, ** default_config):
    def wrapper(dataset_loader):
        """
            Wrapper for EEG datasets.
            The returned dataset is expected to be a `pd.DataFrame` with columns :
                - eeg      : 2-D np.ndarray (the EEG signal)
                - channels : the position's name for the electrodes (list, same length as eeg)
                - rate     : the default EEG sampling rate
                - id       : the subject's id (if not provided, set to the dataset's name)
                - label    : the expected label (the task performed / simulated or the stimuli, ...)
            The wrapper adds the keys :
                - n_channels    : equivalent to len(pos), the number of eeg channels (electrodes)
                - time      : the session's time (equivalent to the signal's length divided by rate)
                
        """
        @timer(name = '{} loading'.format(name))
        def _load_and_process(directory, * args, per_user_label = False, keep_artifacts = False, keep_passive = True, ** kwargs):
            return dataset_loader(directory, * args, ** kwargs)
        
        from datasets.custom_datasets import add_dataset
        
        fn = _load_and_process
        fn.__name__ = dataset_loader.__name__
        fn.__doc__  = dataset_loader.__doc__
        
        add_dataset(name, processing_fn = fn, task = task, ** default_config)
        
        return fn
    return wrapper

def preprocess_tcia_annots(directory, metadata_file = 'metadata.json', overwrite = False, tqdm = lambda x: x, ** kwargs):
    def parse_serie(path, serie_num = -1):
        dirs = sorted(os.listdir(path), key = lambda d: len(os.listdir(os.path.join(path, d))))
        if len(dirs) == 1:
            raise RuntimeError('Unknown annotation type, only 1 directory for serie path {}'.format(path))

        seg_dirs, imgs_dir = dirs[:-1], os.path.join(path, dirs[-1])

        segmentations_infos = []
        for seg_dir in seg_dirs:
            seg_files = os.listdir(os.path.join(path, seg_dir))
            if len(seg_files) != 1:
                raise RuntimeError('{} annotation files for path {}'.format(os.path.join(path, seg_dir)))
            seg_file = os.path.join(path, seg_dir, seg_files[0])
            
            if seg_file not in all_segs_infos:
                with dcm.dcmread(seg_file) as seg:
                    if hasattr(seg, 'StructureSetROISequence'):
                        organs = [struct.ROIName for struct in seg.StructureSetROISequence]
                    elif hasattr(seg, 'SegmentSequence'):
                        organs = [s.SegmentDescription for s in seg.SegmentSequence]
                    else:
                        raise RuntimeError('Unknown annotation sequence name for file {} :\n{}'.format(seg_file, seg))
                    
                    all_segs_infos[seg_file] = {
                        'id'      : str(seg.PatientName),
                        'sex'     : str(seg.PatientSex),
                        'organs'  : organs
                    }

            patient_id  = all_segs_infos[seg_file]['id']
            patient_sex = all_segs_infos[seg_file]['sex']
            organs      = all_segs_infos[seg_file]['organs']
            
            segmentations_infos.append({
                'subject_id'      : patient_id,
                'serie'           : serie_num,
                'segmentation_id' : seg_dir,
                'sex'             : patient_sex,
                'images_dir'      : imgs_dir,
                'nb_images'       : len(os.listdir(imgs_dir)),
                'images'          : sorted([os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir)]),
                'segmentation'    : seg_file,
                'organs'          : organs
            })
        
        return segmentations_infos

    def parse_client(client_dir):
        series = []
        for i, serie_dir in enumerate(os.listdir(client_dir)):
            series.extend(parse_serie(os.path.join(client_dir, serie_dir), i))
        return series
    
    import pydicom as dcm

    data_dir = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if len(data_dir) != 1:
        raise RuntimeError('{} data dirs in {} :\n{}'.format(len(main_dir), directory, '\n'.join(main_dir)))
    data_dir = data_dir[0]
    
    all_segs_infos = {}
    if not overwrite and metadata_file:
        all_segs_infos = load_json(os.path.join(directory, metadata_file))

    metadata = []
    for client_dir in tqdm(os.listdir(data_dir)):
        client_dir = os.path.join(data_dir, client_dir)
        if os.path.isdir(client_dir): metadata.extend(parse_client(client_dir))
    
    if metadata_file: dump_json(filename = os.path.join(directory, metadata_file), data = all_segs_infos, indent = 4)
    
    return pd.DataFrame(metadata)

if os.path.exists(TCIA_DIR):
    for manifest_dir in os.listdir(TCIA_DIR):
        image_seg_dataset_wrapper(
            name      = [d for d in os.listdir(os.path.join(TCIA_DIR, manifest_dir)) if os.path.isdir(os.path.join(TCIA_DIR, manifest_dir, d))][0],
            task      = IMAGE_SEGMENTATION,
            directory = os.path.join(TCIA_DIR, manifest_dir)
        )(preprocess_tcia_annots)
