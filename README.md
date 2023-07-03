# PhD project on Medical Image Segmentation

- **Date** : 2022-2023
- **Author (PhD student)** : Langlois Quentin, UCLouvain, ICTEAM, Belgium.
- **Supervisor** : Jodogne Sébastien, UCLouvain, ICTEAM, Belgium.
- **Key features** : `totalsegmentator` support in tensorflow 2.x with much fewer dependencies !

This project will not be updated in the near future as my PhD project is now on EEG signal analysis. Check [my other repository](https://github.com/Ananas120/phd_eeg) if you are interested in FL or EEG ! :smile:

## Project structure

```bash
├── custom_architectures
│   ├── totalsegmentator_arch.py
│   ├── unet_arch.py
│   └── yolo_arch.py
├── custom_layers
├── custom_train_objects
│   ├── losses
│   │   ├── dice_loss.py        : custom DiceLoss with sparse support
│   │   ├── ge2e_seg_loss.py    : custom loss for segmentation (experiments failed)
│   │   └── yolo_loss.py
│   ├── __init__.py
│   └── history.py
├── datasets
│   ├── custom_datasets
│   │   ├── medical_datasets
│   │   │   ├── __init__.py
│   │   │   ├── eeg_datasets.py         : \*
│   │   │   └── image_seg_datasets.py   : TCIA + TotalSegmentator datasets processing
├── docker
├── evaluations                         : prediction + results for evaluation (cf evaluation.ipynb)
│   ├── converted_totalsegmentator_lowres   : tensorflow convertion of totalsegmentator 3mm
│   │   ├── segmentations                   : directories where predictions are stored
│   │   ├── map.json                        : mapping file {input_file : infos}
│   │   └── results.json                    : multi-label confusion matrix for each input file
│   ├── data
│   ├── totalsegmentator_fullres            : evaluation of the original totalsegmentator full-res model
├── hparams
├── loggers
├── models
│   ├── detection
│   │   ├── __init__.py
│   │   ├── base_detector.py
│   │   ├── east.py
│   │   ├── med_unet.py             : base MedUNet abstract class defining methods + configurations
│   │   ├── med_unet_clusterer.py   : MedUNet sub-class for image segmentation via clustering (failed)
│   │   ├── med_unet_segmentator.py : MedUNet sub-class for image segmentation (softmax)
│   │   └── yolo.py
│   ├── interfaces
├── notebooks
├── pretrained_models               : directories where models are saved
│   └── totalsegmentator_256            : convertion of the totalsegmentator 3mm model
│       ├── eval
│       ├── outputs
│       ├── saving                      : main saving directory (e.g. model architecture / weights)
│       │   ├── checkpoint              : regular tf.train.Checkpoint 
│       │   ├── ckpt-1.data-00000-of-00001
│       │   ├── ckpt-1.index
│       │   ├── config_models.json      : model configuration (e.g. loss / optimizer)
│       │   ├── historique.json         : training history
│       │   └── model.json              : main model architecture (tf.keras.Model.to_json())
│       ├── training-logs
│       │   ├── checkpoints
│       │   └── eval
│       └── config.json             : MedUNet configuration file
├── unitest
├── utils
│   ├── med_utils
│   │   ├── __init__.py
│   │   ├── data_io.py              : I/O utilities for medical image (e.g. nifti read/write)
│   │   ├── label_utils.py          : utilities for label rearranging / mask transform
│   │   ├── metrics.py              : evaluation functions (multi-label confusion matrix + derivated metrics, like dice)
│   │   ├── pre_processing.py       : pre-processing functions (cropping / padding)
│   │   ├── resampling.py           : utilities for volume resampling
│   │   ├── sparse_utils.py         : utilities for sparse mask manipulation
│   │   └── visualize.py            : unused as plot_utils now supports 3D volume plot
│   ├── plot_utils.py           : plot utilities (3D plot is now supported)
├── LICENSE                 : original licence (AGPL v3.0)
├── Makefile
├── README.md               : this file :D
├── README.original.md      : original readme file
├── dataset_processing.ipynb    : dataset format modification (e.g. combining masks, saving in another format (like npz), ...)
├── detection.ipynb
├── evaluation.ipynb            : evaluation notebook (also possible in the main notebook)
├── example_east.ipynb
├── example_med_unet_cluster.ipynb  : experiments for the clustering model
├── example_med_unet_seg.ipynb      : main notebook for medical image segmentation (with pretrained totalsegmentator + training + evaluation)
├── example_yolo.ipynb
├── example_yolo_generator.ipynb
├── requirements.txt
├── test_image.ipynb                : tests for image visualization
├── test_image_seg_clustering.ipynb : tests for the image segmentation via clustering
├── test_performances.ipynb         : optimization tests
├── test_processing.ipynb
├── test_totalsegmentator.ipynb     : totalsegmentator prediction + evaluation (original library)
├── test_totalsegmentator_convertion.ipynb  : tests for the pytorch -> tensorflow convertion
├── test_unet.ipynb                 : UNet architecture experiments
├── test_visualization.ipynb
└── tests.ipynb                     : general tests (e.g. resampling, sparse features, ...)
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

- Main class : `MedUNet` (inherits from `BaseModel`) in the `models/detection/med_unet.py` file
- Main notebook : `example_med_unet_seg.ipynb`, model initialization + instanciation + training + evaluation (+ automatic totalsegmentator convertion)
- Architecture : `custom_architectures/totalsegmentator_arch.py`, original architecture + convertion to tensorflow
- Medical image I/O : `utils/med_utils/data_io.py`, all functions for medical image I/O

## Available models

### Model architectures

Available architectures : 
- [TotalSegmentator: robust segmentation of 104 anatomical structures in CT images](https://arxiv.org/abs/2208.05868) : the original models available in the [totalsegmentator](https://github.com/wasserth/TotalSegmentator) library are fully supported / convertible to tensorflow, without even the library required !

### Model weights

| Classes   | Dataset   | Architecture  | Trainer   | Weights   |
| :-------: | :-------: | :-----------: | :-------: | :-------: |

Models must be unzipped in the `pretrained_models/` directory !

**Important Note** : the official `totalsegmentator` models are fully supported / convertible, and only requires `pytorch` installed (GPU-version **not** required) !

## Installation and usage

1. Clone this repository : `git clone https://github.com/Ananas120/phd_segmentation.git`
2. Go to the root of this repository : `cd phd_segmentation`
3. Install requirements : `pip install -r requirements.txt`
4. Open `example_med_unet_seg` notebook and follow the instructions !

**Important Note** : some *heavy* requirements are removed in order to avoid unnecessary installation of such packages (e.g. `torch`), as they are used only in very specific functions (e.g. weights convertion).  It is therefore possible that some `ImportError` occurs when using specific functions.

## TO-DO list :

- [x] Make the TO-DO list
- [x] Convert totalsegmentator
    - [x] Correctly convert the architecture
    - [x] Transfer weights from pytorch to tensorflow
    - [x] Test the pre-processing convertion
    - [x] Test the post-processing convertion
    - [x] Automatically download pre-trained weights files
    - [x] Remove the `nnunet` / `totalsegmentator` dependencies for convertion
    - [x] Integrate everything with the main `MedUNet` class
    - [x] Perform an evaluation of the original vs converted models to assess equivalence
- [x] Comment the code
- [x] Support Nifti files (read and write)
- [x] Support Dicom Series (read only)
- [x] Support custom (more optimized) saving formats (npz, tensor, stensor)


## Contacts and licence

### My contacts

- PhD student email : quentin.langlois@uclouvain.be
- Supervisor email : sebastien.jodogne@uclouvain.be

### Original contact and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [Affero GPL (AGPL) v3 licence](LICENCE)

All my projects are "free software", meaning that you can use, modify, deploy and distribute them on a free basis, in compliance with the Licence. They are not in the public domain and are copyrighted, there exist some conditions on the distribution but their objective is to make sure that everyone is able to use and share any modified version of these projects. 

Furthermore, if you want to use any project in a closed-source project, or in a commercial project, you will need to obtain another Licence. Please contact me for more information. 

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project or make a Pull Request to solve it :smile: 

If you use this project in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references

Papers :
- [2] [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/abs/1809.10486) : nnU-Net paper
- [2] [TotalSegmentator: robust segmentation of 104 anatomical structures in CT images](https://arxiv.org/abs/2208.05868) : the TotalSegmentator paper

GitHub projects : 
- [nnunet](https://github.com/MIC-DKFZ/nnUNet) : `totalsegmentator` is based on a custom version of `nnunet` (version 1)
- [totalsegmentator](https://github.com/wasserth/TotalSegmentator) : the original totalsegmentator library
- [The detection repo](https://github.com/yui-mhcp/detection) : this repository is a forked version of this original repo
- [The main-classes repo](https://github.com/yui-mhcp/base_dl_project) : repository containing information about the main `BaseModel` class + its main features

Datasets :
- [TotalSegmentator](https://zenodo.org/record/6802614) dataset : the main dataset used for totalsegmentator training / evaluation
