
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
import logging
import scipy.io
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from scipy.signal import resample

from loggers import timer
from utils.file_utils import load_data, dump_data

logger = logging.getLogger(__name__)

PASSIVE_LABELS = ['passive', 'break']

def eeg_dataset_wrapper(name, task, ** default_config):
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
            dataset = dataset_loader(directory, * args, ** kwargs)

            if 'pos' in dataset.columns: dataset.rename(columns = {'pos' : 'channels'}, inplace = True)
            if not keep_passive:
                dataset = dataset[dataset['label'].apply(lambda l: not any(passive_label in l for passive_label in PASSIVE_LABELS))]
            if 'has_artifact' in dataset.columns and not keep_artifacts:
                dataset = dataset[~dataset['has_artifact']]
            if 'time' not in dataset.columns:
                dataset['time'] = dataset.apply(lambda row: row['eeg'].shape[1] / row['rate'], axis = 1)
            if 'n_channels' not in dataset.columns:
                dataset['n_channels'] = dataset['channels'].apply(len)
            if 'id' not in dataset.columns:
                dataset['id'] = name
            else:
                dataset['id'] = dataset['id'].apply(lambda subj_id: '{}-{}'.format(name, subj_id))
            
            if per_user_label:
                dataset['label'] = dataset.apply(lambda row: '{}-{}'.format(row['id'], row['label']), axis = 1)
            
            dataset['dataset_name'] = name
            
            return dataset
        
        from datasets.custom_datasets import add_dataset
        
        fn = _load_and_process
        fn.__name__ = dataset_loader.__name__
        fn.__doc__  = dataset_loader.__doc__
        
        add_dataset(name, processing_fn = fn, task = task, ** default_config)
        
        return fn
    return wrapper

def resample_events(events, target_rate, out_filename, tqdm = lambda x: x):
    from utils.eeg_utils import resample_eeg
    logger.info('Resampling EEG to {}Hz and saving in {}...'.format(
        target_rate, out_filename
    ))
    if isinstance(events, dict): events = events.values()
    for v in tqdm(events):
        if target_rate != v['rate']:
            v.update({
                'rate' : target_rate, 'eeg' : resample_eeg(v['eeg'], rate = v['rate'], target_rate = target_rate)
            })
    if out_filename: dump_data(filename = out_filename, data = events)
    return events

@eeg_dataset_wrapper(
    name  = 'BCI-IV 2a', task = 'BCI',
    train = {'directory' : '{}/BCI-IV/dataset_2/a', 'subset' : 'train'},
    valid = {'directory' : '{}/BCI-IV/dataset_2/a', 'subset' : 'test'}
)
def preprocess_bciiv_2a_annots(directory, subset, subjects = None, target_rate = None, overwrite = False,
                               time_window = 7., tqdm = lambda x: x, ** kwargs):
    _event_names = {
        '276' : 'passive eyes open', '277' : 'passive eyes closed', '768' : 'start trial', '769' : 'left hand (IM)',
        '770' : 'right hand (IM)', '771' : 'feet (IM)', '772' : 'tongue (IM)', '783' : 'passive', '1023' : 'rejected',
        '1072' : 'passive eyes movement', '32766' : 'start run'
    }
    _list_labels = [_event_names[i] for i in ['769', '770', '771', '772']]
    
    _is_processed_dir = not any(d.endswith('gdf') for d in os.listdir(directory))
    
    dataset = []
    for file in os.listdir(directory):
        if not (file.endswith('gdf') or (_is_processed_dir and file.endswith('mat'))): continue
        elif subset == 'train' and 'T' not in file: continue
        elif subset == 'test' and 'T' in file: continue
        elif subjects and file[2] not in subjects: continue
        
        filename = os.path.join(directory, file)
        resampled_filename  = filename.replace(
            '.gdf' if not _is_processed_dir else '.mat', '_{}.pkl'.format(target_rate)
        )
        
        if target_rate and os.path.exists(resampled_filename) and not overwrite:
            logger.debug('Loading pickled resampled EEG from {}'.format(resampled_filename))
            dataset.extend(load_data(resampled_filename))
            continue
        
        file_datas = []
        if _is_processed_dir:
            data = scipy.io.loadmat(filename)['data']

            for i in range(data.size):
                data_1 = data[0, i]
                data_2 = data_1[0, 0]

                eeg, starts, y, fs, labels, artifacts, gender, age = [np.squeeze(d) for d in data_2]
                eeg = eeg.T[:22]
                for trial in range(starts.size):
                    start, end = starts[trial], starts[trial] + int(time_window * 250)
                    file_datas.append({
                        'id'    : file[2],
                        'age'   : age,
                        'device'    : '?',
                        'gender'    : gender,
                        'has_artifact' : artifacts[trial] != 0,
                        'label' : labels[y[trial] - 1][0] + ' (IM)',
                        'rate'  : 250,
                        'time'  : (end - start) / 250,
                        'channels'   : ['EEG-{}'.format(i) for i in range(22)],
                        'eeg'   : eeg[:, start : end]
                    })
        else:
            import mne
            
            data   = mne.io.read_raw_gdf(filename)
            events, event_ids = mne.events_from_annotations(data)

            target = None
            if subset == 'test' and os.path.exists(filename.replace('.gdf', '.mat')):
                target = scipy.io.loadmat(filename.replace('gdf', 'mat'))['classlabel'][:, 0]

            _id_to_event = {v : k for k, v in event_ids.items()}

            eeg    = data.get_data()[:-3, :]
            pos    = [name if name.split('-')[1].isdigit() else name.split('-')[1] for name in data.ch_names[:-3]]

            n_runs, n_trials, trial_start, reject = 1, 0, 0, False
            for i, (start, _, event) in enumerate(events):
                event_name = _event_names[_id_to_event[event]]
                if 'run' in event_name and n_trials > 0: n_runs += 1
                elif 'trial' in event_name: n_trials += 1
                if 'start' in event_name or 'reject' in event_name:
                    trial_start = start
                    reject = False if 'start' in event_name else True
                    continue

                file_datas.append({
                    'id'    : file[2],
                    'device'    : '?',
                    'label' : event_name if target is None or n_trials == 0 else _list_labels[target[n_trials - 1] - 1],
                    'has_artifact' : reject,
                    'rate'  : 250,
                    'time'  : time_window,
                    'channels'   : pos,
                    'eeg'   : eeg[:, trial_start : trial_start + int(time_window * 250)]
                })
            assert n_trials == 288 and n_runs == 6, 'Got {} runs and {} trials which is unexpected !'.format(n_runs, n_trials)
    
        if target_rate:
            resample_events(file_datas, target_rate, out_filename = resampled_filename, tqdm = tqdm)

        dataset.extend(file_datas)
    
    return pd.DataFrame(dataset)

    
@eeg_dataset_wrapper(name = 'Brain MNIST', task = 'BCI', directory = '{}/Brain_MNIST')
def preprocess_brain_mnist_annots(directory, subset = None, target_rate = None,
                                  tqdm = lambda x: x, ** kwargs):
    _rates = {'IN' : 128, 'MU' : 220, 'EP' : 128, 'MW' : 512}
    if isinstance(subset, str): subset = [subset]
    
    events = {}
    for f in glob.glob(os.path.join(directory, '*.txt')):
        if subset and not any(sub in f for sub in subset): continue

        resampled_filename = f.replace('.txt', '_{}.pkl'.format(target_rate))
        if target_rate and os.path.exists(resampled_filename):
            logger.debug('Loading pickled resampled EEG from {}'.format(resampled_filename))
            events.update(load_data(resampled_filename))
            continue
            
        with open(f, 'r', encoding = 'utf-8') as file:
            lines = file.read().split('\n')
        
        file_events = {}
        for l in tqdm(lines):
            if not l: continue
            
            data_id, event, device, pos, label, size, eeg = l.split('\t')
            size, label, eeg = int(size), int(label), [float(v) for v in eeg.split(',')]
            
            file_events.setdefault(event, {
                'id'    : '1',
                'device'    : device,
                'label' : label,
                'rate'  : _rates[device],
                'time'  : size / _rates[device],
                'channels' : [],
                'eeg'      : []
            })
            file_events[event]['eeg'].append(eeg)
            file_events[event]['channels'].append(pos)
        
        for k, v in file_events.items(): v['eeg'] = np.array(v['eeg'], dtype = np.float32)
        
        if target_rate:
            resample_events(file_events, target_rate, out_filename = resampled_filename, tqdm = tqdm)
        
        events.update(file_events)
    
    return pd.DataFrame(list(events.values()))

@eeg_dataset_wrapper(name = 'EEGMMIDB', task = 'BCI', directory = '{}/EEGMMIDB')
def preprocess_eegmmidb_annots(directory, subjects = None, target_rate = None, overwrite = False,
                               tqdm = lambda x: x, ** kwargs):
    _labels = {
        0 : 'open eyes', 1 : 'close eyes', 2 : 'left hand', 3 : 'right hand',
        4 : 'left hand (IM)', 5 : 'right hand (IM)', 6 : 'both fists', 7 : 'both feet',
        8 : 'both fists (IM)', 9 : 'both feet (IM)', 10 : 'passive'
    }
    _pos = [
        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
        'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ',
        'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
        'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ'
    ]
    
    if subjects is not None: subjects = list(subjects)
    else: subjects = list(range(1, 110))
    
    dataset = []
    for subj_id in tqdm(subjects):
        filename = os.path.join(directory, '{}.npy'.format(subj_id))
        resampled_filename = filename.replace('.npy', '_{}.pkl'.format(target_rate))
        if not overwrite and target_rate and os.path.exists(resampled_filename):
            logger.debug('Loading pickled resampled EEG from {}'.format(resampled_filename))
            dataset.extend(load_data(resampled_filename))
            continue
        
        data = np.load(filename).T.astype(np.float32)
        
        eeg, label = data[:-1], data[-1]
        from utils import plot_audio
        
        plot_audio(audio = label[120 * 160 : 240 * 160], rate = 160)
        
        groups = []
        current_idx, file_datas = 0, []
        for label, group in itertools.groupby(label):
            group = list(group)
            groups.append(label)
            
            file_datas.append({
                'id'    : subj_id,
                'device'    : 'BCI 2000',
                'label' : _labels[label],
                'rate'  : 160,
                'time'  : len(group) / 160,
                'channels' : _pos,
                'eeg'   : eeg[:, current_idx : current_idx + len(group)]
            })
            current_idx += len(group)
    
        print(np.bincount(groups))
        if target_rate:
            resample_events(file_datas, target_rate, out_filename = resampled_filename, tqdm = tqdm)

        dataset.extend(file_datas)
    
    return pd.DataFrame(dataset)

@eeg_dataset_wrapper(name = 'Large scale BCI', task = 'BCI', directory = '{}/Large_scale_BCI')
def process_large_scale_bci_annots(directory, subset = None, subjects = None, target_rate = None,
                                   tqdm = lambda x: x, overwrite = False, ** kwargs):
    _cla_marker_label   = {
        0 : 'passive', 1 : 'left hand (IM)', 2 : 'right hand (IM)', 3 : 'neutral', 4 : 'left leg (IM)',
        5 : 'tongue (IM)', 6 : 'right leg (IM)', 91 : 'break', 99 : 'relaxation'
    }
    _5f_marker_label   = {
        0 : 'passive', 1 : 'thumb finger (IM)', 2 : 'index finger (IM)', 3 : 'middle finger (IM)',
        4 : 'ring finger (IM)', 5 : 'pinkie finger (IM)', 91 : 'break', 99 : 'relaxation'
    }
    
    if subset is not None:
        if not isinstance(subset, (list, tuple)): subset = [subset]
        subset = [s.lower() for s in subset]
    
    dataset = []
    for file in tqdm(os.listdir(directory)):
        if not file.endswith('.mat'): continue

        if '-' in file:
            task, subj_id, date = file.split('-')[:3]
        else:
            subj_idx, n = file.index('Subject'), len('Subject')
            task, subj_id, date = file[:subj_idx], file[subj_idx : subj_idx + n + 1], file[subj_idx + n + 1 : subj_idx + n + 7]
        subj_id = subj_id[-1:]
        task = task.lower()
        
        if subset and task not in subset: continue
        if subjects is not None and subj_id not in subjects: continue
        
        filename = os.path.join(directory, file)
        resampled_filename  = filename.replace('.mat', '_{}.pkl'.format(target_rate))
        
        if target_rate and os.path.exists(resampled_filename) and not overwrite:
            logger.debug('Loading pickled resampled EEG from {}'.format(resampled_filename))
            data = load_data(resampled_filename)
            dataset.extend(data)
            continue
        
        data = scipy.io.loadmat(filename)['o'][0, 0]

        data_id, freq, n_samp, marker, eeg, pos = [np.squeeze(data[i]) for i in [0, 2, 3, -4, -3, -2]]
        freq, n_samp = min(n_samp, freq), max(n_samp, freq)
        eeg = eeg.T[:-1].astype(np.float32) # last channel is a synchronization channel, not EEG signal
        pos = [p[0] for p in pos][:len(eeg)]

        _marker_to_label    = _5f_marker_label if task == '5F' else _cla_marker_label
        
        file_datas  = []
        
        current_idx, started = 0, False
        for label, group in itertools.groupby(marker):
            if label == 91: started = True
            elif label == 92: break
            
            group = list(group)
            if not started:
                current_idx += len(group)
                continue
            
            file_datas.append({
                'id'    : subj_id,
                'device'    : 'EEG 1200',
                'task'  : task,
                'label' : _marker_to_label[label],
                'rate'  : freq,
                'time'  : len(group) / freq,
                'channels' : pos,
                'eeg'   : eeg[:, current_idx : current_idx + len(group)]
            })
            current_idx += len(group)
    
        if target_rate:
            resample_events(file_datas, target_rate, out_filename = resampled_filename, tqdm = tqdm)

        dataset.extend(file_datas)
    
    return pd.DataFrame(dataset)

@eeg_dataset_wrapper(name = 'SEED', task = 'EEG emotion recognition', directory = '{}/SEED-V')
def process_seed_annots(directory, subset = None, subjects = None, tqdm = lambda x: x,
                        ** kwargs):
    import mne
    
    _session_timestamps = {
        1 : [
            [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
            [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]
        ],
        2 : [
            [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
            [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]
        ],
        3 : [
            [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
            [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
        ]
    }
    _labels = {0 : 'disgust', 1 : 'fear', 2 : 'sad', 3 : 'neutral', 4 : 'happy'}
    
    session_label   = load_data(os.path.join(directory, 'emotion_label_and_stimuli_order.xlsx'))
    session_label   = session_label.iloc[-3:, 2:].values
    
    if subset is not None and not isinstance(subset, (list, tuple)): subset = [subset]
    if subjects is not None and not isinstance(subjects, (list, tuple)): subjects = [subjects]

    rate        = 1000

    dataset = []
    for file in tqdm(os.listdir(os.path.join(directory, 'EEG_raw'))):
        if not file.endswith('.cnt'): continue

        subj_id     = int(file.split('_')[0])
        session     = int(file.split('_')[1])
        
        if subset and session not in subset: continue
        if subjects and subj_id not in subjects: continue
        
        eeg_file    = mne.io.read_raw_cnt('D:/datasets/SEED-V/EEG_raw/6_3_20180802.cnt')
        eeg         = eeg_file.get_data()

        for i, (start, end) in enumerate(zip(* _session_timestamps[session])):
            dataset.append({
                'id'    : subj_id,
                'label' : session_label[session - 1, i],
                'rate'  : rate,
                'size'  : (end - start) * rate,
                'time'  : end - start,
                'pos'   : eeg_file.ch_names,
                'eeg'   : eeg[:, start * rate : end * rate]
            })
    
    return pd.DataFrame(dataset)
