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
from utils.med_utils import get_nb_frames
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
        def _load_and_process(directory, * args, slice_step = None, slice_size = None, ** kwargs):
            dataset = dataset_loader(directory, * args, ** kwargs)
            
            if slice_step or slice_size:
                if not slice_step: slice_step = slice_size
                
                augmented = []
                for data in dataset:
                    for start in range(0, get_nb_frames(data), slice_step):
                        augmented.append(data.copy())
                        augmented[-1]['start_frame'] = start
                        if slice_size: augmented[-1]['end_frame'] = start + slice_size
                dataset = augmented
            
            dataset = pd.DataFrame(dataset)
            
            if 'id' not in dataset.columns:
                dataset['id'] = dataset['subject_id']

            return dataset
        
        from datasets.custom_datasets import add_dataset
        
        fn = _load_and_process
        fn.__name__ = dataset_loader.__name__
        fn.__doc__  = dataset_loader.__doc__
        
        add_dataset(name, processing_fn = fn, task = task, ** default_config)
        
        return fn
    return wrapper

def preprocess_tcia_annots(directory,
                           subset = None,
                           
                           overwrite     = False,
                           metadata_file = 'metadata.json',
                           
                           tqdm = lambda x: x,
                           ** kwargs
                          ):
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
                    #elif hasattr(seg, 'SegmentSequence'):
                    #    organs = [s.SegmentDescription for s in seg.SegmentSequence]
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
            
            frames      = sorted([os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir)])
            with dcm.dcmread(frames[0]) as ct1, dcm.dcmread(frames[-1]) as ct2:
                if ct1.SliceLocation > ct2.SliceLocation: frames = frames[::-1]
                thickness = ct1.SliceThickness
            
            segmentations_infos.append({
                'subject_id'      : patient_id,
                'serie'           : serie_num,
                'segmentation_id' : seg_dir,
                'sex'             : patient_sex,
                'thickness'       : thickness,
                'images_dir'      : imgs_dir,
                'nb_images'       : len(os.listdir(imgs_dir)),
                'images'          : frames,
                'segmentation'    : seg_file,
                'labels'          : organs
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
        if subset and subset not in client_dir: continue
        client_dir = os.path.join(data_dir, client_dir)
        if os.path.isdir(client_dir): metadata.extend(parse_client(client_dir))
    
    if metadata_file: dump_json(filename = os.path.join(directory, metadata_file), data = all_segs_infos, indent = 4)
    
    return metadata

@image_seg_dataset_wrapper(
    name  = 'total_segmentator', task  = IMAGE_SEGMENTATION, directory = '{}/Totalsegmentator_dataset'
)
def preprocess_totalsegmentator_annots(directory, combined_mask = ('masks.npz', 'masks.nii.gz'), tqdm = lambda x: x, ** kwargs):
    metadata = []
    for client_id in tqdm(os.listdir(directory)):
        if '.' in client_id: continue
        client_dir = os.path.join(directory, client_id)
        
        segmentations = list(sorted(
            (filename.split('.')[0], os.path.join(client_dir, 'segmentations', filename))
            for filename in os.listdir(os.path.join(client_dir, 'segmentations'))
        ))
        
        mask_file = None
        for comb_mask_file in combined_mask:
            comb_mask_file = os.path.join(client_dir, comb_mask_file)
            if os.path.exists(comb_mask_file):
                mask_file = comb_mask_file
                break
        
        metadata.append({
            'subject_id'      : client_id,
            'thickness'       : -1,
            'images'          : os.path.join(client_dir, 'ct.nii.gz'),
            'segmentation'    : [file for _, file in segmentations] if not mask_file else mask_file,
            'label'           : [organ for organ, _ in segmentations]
        })
    
    return metadata

tcia_manifests = {
    'LSTCS'        : 'manifest-1557326747206',
    'Radiomics'    : 'manifest-1603198545583',
    'Pediatric-CT' : 'manifest-1647979711903'
}

if os.path.exists(TCIA_DIR):
    image_seg_dataset_wrapper(
        name  = 'LSTCS',
        task  = IMAGE_SEGMENTATION,
        train = {'directory' : os.path.join(TCIA_DIR, 'manifest-1557326747206'), 'subset' : 'Train'},
        valid = {'directory' : os.path.join(TCIA_DIR, 'manifest-1557326747206'), 'subset' : 'Test'}
    )(preprocess_tcia_annots)

    image_seg_dataset_wrapper(
        name      = 'Radiomics',
        task      = IMAGE_SEGMENTATION,
        directory = os.path.join(TCIA_DIR, 'manifest-1603198545583')
    )(preprocess_tcia_annots)

    image_seg_dataset_wrapper(
        name      = 'Pediatric-CT',
        task      = IMAGE_SEGMENTATION,
        directory = os.path.join(TCIA_DIR, 'manifest-1647979711903')
    )(preprocess_tcia_annots)

"""
if os.path.exists(TCIA_DIR):
    for manifest_dir in os.listdir(TCIA_DIR):
        image_seg_dataset_wrapper(
            name      = [d for d in os.listdir(os.path.join(TCIA_DIR, manifest_dir)) if os.path.isdir(os.path.join(TCIA_DIR, manifest_dir, d))][0],
            task      = IMAGE_SEGMENTATION,
            directory = os.path.join(TCIA_DIR, manifest_dir)
        )(preprocess_tcia_annots)
"""