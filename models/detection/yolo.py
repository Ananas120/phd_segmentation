
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
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
import time
import shutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils.thread_utils import Pipeline, Consumer
from custom_architectures import get_architecture
from models.interfaces.base_image_model import BaseImageModel
from utils import load_json, dump_json, normalize_filename, download_file, plot
from utils.image import _video_formats, _image_formats, load_image, save_image, stream_camera, BASE_COLORS, get_video_infos
from utils.image.box_utils import *

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

PRETRAINED_COCO_URL = 'https://pjreddie.com/media/files/yolov2.weights'

DEFAULT_ANCHORS = [
    0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
]

COCO_CONFIG = {
    'labels' : ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'brocoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
    'anchors' : DEFAULT_ANCHORS
}
VOC_CONFIG  = {
    'labels' : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    'anchors'   : DEFAULT_ANCHORS #[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
}

class YOLO(BaseImageModel):
    get_input   = BaseImageModel.get_image
    augment_input   = BaseImageModel.augment_image
    preprocess_input    = BaseImageModel.preprocess_image
    
    def __init__(self,
                 labels, 
                 max_box_per_image  = 100,
                 
                 input_size = 416,
                 nb_class   = None,
                 backend    = "FullYolo",
                 
                 anchors    = DEFAULT_ANCHORS,

                 obj_threshold  = 0.35,
                 nms_threshold  = 0.2,

                 **kwargs
                ):
        assert len(anchors) % 2 == 0

        self._init_image(input_size = input_size, ** kwargs)
        
        self.backend    = backend
        self.anchors    = anchors
        self.max_box_per_image = max_box_per_image

        self.labels   = list(labels) if not isinstance(labels, str) else [labels]
        self.nb_class = max(2, nb_class if nb_class is not None else len(self.labels))
        if self.nb_class > len(self.labels):
            self.labels += [''] * (self.nb_class - len(self.labels))
        
        self.obj_threshold  = obj_threshold
        self.nms_threshold  = nms_threshold
        
        self.np_anchors = np.reshape(np.array(anchors), [self.nb_box, 2])

        super().__init__(** kwargs)

    def init_train_config(self, * args, ** kwargs):
        super().init_train_config(* args, ** kwargs)
        
        if hasattr(self, 'model_loss'):
            self.model_loss.seen = self.current_epoch
            
    def _build_model(self, flatten = True, randomize = True, ** kwargs):
        feature_extractor = get_architecture(
            architecture_name = self.backend,
            input_shape = self.input_size,
            include_top = False,
            ** kwargs
        )
        
        super()._build_model(model = {
            'architecture_name' : 'yolo',
            'feature_extractor' : feature_extractor,
            'input_shape'       : self.input_size,
            'nb_class'      : self.nb_class,
            'nb_box'        : self.nb_box,
            'flatten'       : flatten,
            'randomize'     : randomize
        })
    
    @property
    def stream_dir(self):
        return os.path.join(self.folder, "stream")
        
    @property
    def output_signature(self):
        return (
            tf.TensorSpec(
                shape = (None, self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class),
                dtype = tf.float32
            ),
            tf.TensorSpec(
                shape = (None, 1, 1, 1, self.max_box_per_image, 4), dtype = tf.float32
            )
        )
    
    @property
    def nb_box(self):
        return len(self.anchors) // 2
    
    @property
    def grid_h(self):
        return self.output_shape[1]
    
    @property
    def grid_w(self):
        return self.output_shape[2]
        
    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_image)
    
    def __str__(self):
        des = super().__str__()
        des += self._str_image()
        des += "- Labels (n = {}) : {}\n".format(len(self.labels), self.labels)
        des += "- Feature extractor : {}\n".format(self.backend)
        return des
    
    @timer(name = 'inference', log_if_root = False)
    def detect(self, image, get_boxes = False, training = False, ** kwargs):
        """
            Performs prediction on `image` and returns either the model's output either the boxes (if `get_boxes = True`)
            
            Arguments :
                - image : tf.Tensor of rank 3 or 4 (single / batched image(s))
                - get_boxes : bool, whether to decode the model's output or not
                - training  : whether to make prediction in training mode
                - kwargs    : forwarded to `decode_output` if `get_boxes = True`
            Return :
                if `get_boxes == False` :
                    model's output of shape (B, grid_h, grid_w, nb_box, 5 + nb_class)
                else:
                    list of boxes (where boxes is the list of BoundingBox for detected objects)
                
        """
        if not isinstance(image, tf.Tensor): image = tf.cast(image, tf.float32)
        if len(tf.shape(image)) == 3:        image = tf.expand_dims(image, axis = 0)
        
        outputs = self(image, training = training)
        
        if not get_boxes: return outputs
        
        return [self.decode_output(out, ** kwargs) for out in outputs]
    
    def compile(self, loss = 'YoloLoss', loss_config = {}, ** kwargs):
        loss_config.update({'anchors' : self.anchors})
        
        super().compile(loss = loss, loss_config = loss_config, ** kwargs)
    
    @timer(name = 'output decoding')
    def decode_output(self, output, obj_threshold = None, nms_threshold = None, ** kwargs):
        if obj_threshold is None: obj_threshold = self.obj_threshold
        if nms_threshold is None: nms_threshold = self.nms_threshold
        
        grid_h, grid_w, nb_box = output.shape[:3]
        nb_class = output.shape[3] - 5
        time_logger.start_timer('preprocess')

        # decode the output by the network
        pos     = output[..., :4].numpy()
        conf    = tf.sigmoid(output[..., 4:5]).numpy()
        classes = tf.nn.softmax(output[..., 5:], axis = -1).numpy()
        
        scores  = conf * classes
        scores[scores <= obj_threshold] = 0.
        
        class_scores = np.sum(scores, axis = -1)

        conf    = conf[..., 0]
        candidates  = np.where(class_scores > 0.)
        
        time_logger.stop_timer('preprocess')
        time_logger.start_timer('box filtering')

        pos     = pos[candidates]
        conf    = conf[candidates]
        classes = classes[candidates]
        
        row, col, box = candidates
        x, y, w, h    = [pos[:, i] for i in range(4)]
        
        np_anchors = np.array(self.anchors)
        
        x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
        y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
        w = np_anchors[2 * box + 0] * np.exp(w) / grid_w # unit: image width
        h = np_anchors[2 * box + 1] * np.exp(h) / grid_h # unit: image height
        
        x1 = np.maximum(0., x - w / 2.)
        y1 = np.maximum(0., y - h / 2.)
        x2 = np.minimum(1., x + w / 2.)
        y2 = np.minimum(1., y + h / 2.)
        
        valids = np.logical_and(x1 < x2, y1 < y2)
        boxes  = [BoundingBox(
            x1 = float(x1i), y1 = float(y1i), x2 = float(x2i), y2 = float(y2i),
            conf = c, classes = cl
        ) for x1i, y1i, x2i, y2i, c, cl in zip(
            x1[valids], y1[valids], x2[valids], y2[valids], conf[valids], classes[valids]
        )]

        time_logger.stop_timer('box filtering')
        time_logger.start_timer('NMS')
        # suppress non-maximal boxes
        ious = {}
        for c in range(nb_class):
            scores = np.array([box.classes[c] for box in boxes])
            sorted_indices = np.argsort(scores)[::-1]
            sorted_indices = sorted_indices[scores[sorted_indices] > 0]

            for i, index_i in enumerate(sorted_indices):
                if boxes[index_i].classes[c] == 0: continue
                
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if boxes[index_j].classes[c] == 0: continue

                    if (index_i, index_j) not in ious:
                        ious[(index_i, index_j)] = bbox_iou(boxes[index_i], boxes[index_j])
                    
                    if ious[(index_i, index_j)] >= nms_threshold:
                        boxes[index_j].classes[c] = 0
        
        time_logger.stop_timer('NMS')
        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.score > 0]
        return boxes

    def get_output_fn(self, boxes, labels, nb_box, image_h, image_w, ** kwargs):
        if hasattr(boxes, 'numpy'): boxes = boxes.numpy()
        if hasattr(image_h, 'numpy'): image_h, image_w = image_h.numpy(), image_w.numpy()
        
        output      = np.zeros((self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class))
        true_boxes  = np.zeros((1, 1, 1, self.max_box_per_image, 4))
        
        logger.debug("Image with shape ({}, {}) and {} boxes :".format(image_h, image_w, len(boxes)))
        
        for i in range(nb_box):
            x, y, w, h = boxes[i]
            label_idx = self.labels.index(labels[i]) if labels[i] in self.labels else 0

            center_y = ((y + 0.5 * h) / image_h) * self.grid_h
            center_x = ((x + 0.5 * w) / image_w) * self.grid_w
            
            w = (w / image_w) * self.grid_w  # unit: grid cell
            h = (h / image_h) * self.grid_h  # unit: grid cell
            
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            
            logger.debug("Boxes {} ({}) go to grid ({}, {})".format(i, boxes[i], grid_y, grid_x))
            
            if w > 0. and h > 0. and grid_x < self.grid_w and grid_y < self.grid_h:
                box = np.array([center_x, center_y, w, h])
                yolo_box = np.array([center_x, center_y, w, h, 1.])
                
                true_boxes[0, 0, 0, i % self.max_box_per_image, :] = box
                
                # find the anchor that best predicts this box
                
                box_wh = np.repeat(np.array([[w, h]]), self.nb_box, axis = 0)
                
                intersect = np.minimum(box_wh, self.np_anchors)
                intersect = intersect[:,0] * intersect[:,1]
                union = (self.np_anchors[:,0] * self.np_anchors[:,1]) +  (box_wh[:,0] * box_wh[:,1]) - intersect
                
                iou = intersect / union
                
                best_anchor = np.argmax(iou)
                
                logger.debug("Normalized box {} with label {} to anchor idx {} with score {}".format(box, label_idx, best_anchor, iou[best_anchor]))
                
                if iou[best_anchor] > 0.:
                    output[grid_y, grid_x, best_anchor, :5] = yolo_box
                    output[grid_y, grid_x, best_anchor, 5 + label_idx] = 1
        
        return output, true_boxes
    
    def get_output(self, infos):
        output, true_boxes = tf.py_function(
            self.get_output_fn,
            [infos['box'], infos['label'], infos['nb_box'], infos['height'], infos['width']],
            Tout = [tf.float32, tf.float32]
        )

        output.set_shape([self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class])
        true_boxes.set_shape([1, 1, 1, self.max_box_per_image, 4])
        
        return output, true_boxes
    
    def encode_data(self, data):
        return self.get_input(data), self.get_output(data)
    
    def augment_data(self, image, output):
        return self.augment_input(image), output
    
    def preprocess_data(self, image, output):
        return self.preprocess_input(image), output
    
    @timer(name = 'drawing')
    def draw_prediction(self, image, boxes, labels = None, as_mask = False, ** kwargs):
        """ Calls `draw_boxes` or `mask_boxes` depending on `as_mask` and returns the result """
        if len(boxes) == 0: return image
        
        if as_mask:
            return mask_boxes(image, boxes, ** kwargs)
        
        kwargs.setdefault('color', BASE_COLORS)
        kwargs.setdefault('use_label', True)
        kwargs.setdefault('labels', labels if labels is not None else self.labels)

        return draw_boxes(image, boxes, ** kwargs)
    
    @timer
    def save_image(self, image, directory = None, filename = 'image_{}.jpg', ** kwargs):
        """ Saves `image` to `directory` (default `{self.pred_dir}/images/`) and returns its path """
        if isinstance(image, str): return image
        if directory is None: directory = os.path.join(self.pred_dir, 'images')
        
        if not filename.startswith(directory): filename = os.path.join(directory, filename)
        if '{}' in filename:
            filename = filename.format(len(glob.glob(filename.replace('{}', '*'))))
        
        save_image(filename = filename, image = image)
        
        return filename
    
    @timer
    def save_detected(self,
                      detected  = None,
                      image = None,
                      boxes = None,
                      dezoom_factor = 1.,
                      
                      directory = None,
                      filename  = '{}_detected.jpg',
                      original_filename = None,
                      
                      ** kwargs
                     ):
        """
            Saves the image with drawn detection in `directory` (default `{self.pred_dir}/detected`)
            
            Arguments :
                - detected  : the image with already drawn predictions
                - image / boxes / dezoom_factor : information required to produce `detected` if not provided. This information is required if `detected` is not provided, ignored otherwise
                
                - directory : where to save the image
                - filename  : the filename format of the image
                - original_filename : the filename of the original image
                
                - kwargs    : forwarded to `self.draw_prediction`
            Return :
                - filename  : the filename of the saved image
        """
        if detected is None:
            assert boxes is not None and image is not None, 'You must provide either `detected` either `image + boxes`'
            
            image_h, image_w = get_image_size(image)
            normalized_boxes = [get_box_pos(
                box, image_h = image_h, image_w = image_w, labels = labels,
                with_label = True, dezoom_factor = dezoom_factor,
                normalize_mode = NORMALIZE_WH
            )[:5] for box in boxes]
            
            detected = self.draw_prediction(image, boxes, ** kwargs)

        if directory is None: directory = os.path.join(directory, 'detected')
        if not filename.startswith(directory): filename = os.path.join(directory, filename)
        
        if '{}' in filename:
            if isinstance(original_filename, str):
                img_name    = os.path.splitext(os.path.basename(original_filename))[0]
            else:
                img_name    = 'image_{}'.format(len(glob.glob(filename.replace('{}', '*'))))
            
            filename = filename.format(img_name)
        
        save_image(filename = filename, image = detected)
        
        return filename

    def _get_saving_functions(self, max_workers = -1, ** kwargs):
        fake_fn = lambda * args, ** kwargs: None
        
        saving_functions    = [
            kwargs.get('show_result_fn',  None),
            kwargs.get('save_json_fn',    dump_json),
            kwargs.get('save_image_fn',   self.save_image),
            kwargs.get('save_detected_fn', self.save_detected),
            kwargs.get('save_boxes_fn',   extract_boxes)
        ]
        
        if max_workers >= 0:
            for i in range(len(saving_functions)):
                if saving_functions[i] is not None:
                    saving_functions[i] = Consumer(
                        saving_functions[i], max_workers = max_workers if i > 1 else 0
                    )
                    saving_functions[i].start()
        
        return [
            fn if fn is not None else fake_fn for fn in saving_functions
        ]
    
    @timer
    def stream(self, stream_name = 'stream_{}', save = False, max_workers = 0, ** kwargs):
        """
            Performs streaming either on camera (default) or on filename (by specifying the `cam_id` kwarg)
        """
        kwargs.update({'save' : save})
        if stream_name:
            directory = kwargs.get('directory', self.stream_dir)
            if '{}' in stream_name:
                stream_name = stream_name.format(len(
                    glob.glob(os.path.join(directory, stream_name.replace('{}', '*')))
                ))
            
            kwargs.update({
                'directory' : os.path.join(directory, stream_name),
                'raw_img_dir'   : os.path.join(directory, stream_name, 'frames'),
                'detected_dir'  : os.path.join(directory, stream_name, 'detected'),
                'boxes_dir'     : os.path.join(directory, stream_name, 'boxes')
            })
        else:
            kwargs.setdefault('directory', self.stream_dir)
        
        # for tensorflow-graph compilation (the 1st call is much slower than the next ones)
        self.detect(tf.random.uniform(self.input_size))

        saving_functions    = self._get_saving_functions(
            max_workers = max_workers, show_result_fn = None, save_json_fn = None
        )
        
        map_file    = os.path.join(kwargs['directory'], 'map.json')
        predicted   = load_json(map_file, default = {})
        
        stream_camera(
            transform_fn     = lambda img: self.predict(
                img,
                force_draw  = True,
                predicted   = predicted,
                create_dirs = img['frame_index'] == 0,
                saving_functions    = saving_functions,
                ** kwargs
            )[0][1],
            max_workers = max_workers,
            add_copy    = True,
            add_index   = True,
            name        = 'frame transform',
            ** kwargs
        )
        
        for fn in saving_functions:
            if isinstance(fn, Consumer): fn.join()
        
        if predicted: dump_json(map_file, predicted, indent = 4)
        
        return map_file
    
    @timer
    def predict(self,
                images,
                batch_size = 16,
                # general saving config
                directory   = None,
                overwrite   = False,
                timestamp   = -1,
                max_workers = -1,
                create_dirs = True,
                saving_functions    = None,
                # Saving mapping + raw images
                save    = True,
                predicted   = None,
                img_num = -1,
                save_empty  = False,
                raw_img_dir = None,
                filename    = 'image_{}.jpg',
                # Saving images with drawn detection
                force_draw      = False,
                save_detected   = False,
                detected_dir    = None,
                detected_filename   = 'detected_{}.jpg',
                # Saving boxes as individual images
                save_boxes  = False,
                boxes_dir   = None,
                boxes_filename  = '{}_box_{}.jpg',
                # Verbosity config
                verbose = 1,
                display = False,
                plot_kwargs = {},
                
                post_processing = None,
                
                ** kwargs
               ):
        """
            Performs image object detection on the givan `images` (either filename / embeddings / raw)
            
            Arguments :
                - images  : the image(s) to detect objects on
                    - str   : the filename of the image
                    - dict / pd.Series  : informations about the image
                        - must contain at least `filename` or `embedding` or `image_embedding`
                    - np.ndarray / tf.Tensor    : the embedding for the image
                    
                    - list / pd.DataFrame   : an iterable of the above types
                - batch_size    : the number of prediction to perform in parallel
                
                - directory : where to save the mapping file / folders for the saved results
                              if not provided, default to `self.pred_dir`
                - overwrite : whether to overwrite already predicted files (ignored for raw images)
                - timestamp : the timestamp of the request (if `overwrite = True` but this timestamp is lower than the timestamp of the prediction, the image is not overwritte), may be useful for versioning
                - max_workers   : the number of saving workers
                    - -1    : the saving functions are applied sequentially in the main thread
                    - 0     : each saving function is called in a separated thread
                    - >0    : each saving function is called in `max_workers` parallel threads
                - create_dirs   : whether to create sub-folder or not (for streaming optimization)
                - saving_functions  : tuple of 5 elements, the saving functions (for streaming)
                
                - save  : whether to save the results or not (set to `True` if any other `save_*` is True)
                - predicted : the saved mapping (for streaming optimization)
                - img_num   : the image number to use for raw images (for streaming optimization)
                - save_empty    : whether to save result for images without any detected object
                - filename  : filename format for raw images
                
                - force_draw    : whether to force drawing boxes or not (drawing is done by default if `save_detected` or `verbose` or `post_processing` is provided)
                - save_detected : whether to save images with drawn boxes
                - detected_dir  : where to save the detections (default to `{directory}/detected`)
                - detected_filename : filename format for the images with detection
                
                - save_boxes    : whether to save each box as an image
                - boxes_dir     : where to save the extracted boxes (default to `{directory}/boxes`)
                - boxes_filename    : the box filename format
                
                - verbose   : the verbosity level
                    - 0 : silent (no display)
                    - 1 : plots the detected image
                    - 2 : plots the detected image + each individual box
                    - 3 : plots the detected image + each individual box + logs their information
                - display   : number of images to plots (cf `verbose`), if `True` or `-1`, displays all images
                - plot_kwargs   : kwargs for the `plot` calls (ignored if silent mode)
                
                - post_processing   : a callable that takes the detected image as 1st arg + `image` and `infos` as kwargs
                
                - kwargs    : forwarded to `self.infer` (and thus to `self.decode_output`)
            Returns :
                - result    : a list of tuple (image, detected, result)
                    - image     : either the filename (if any), either the original image
                    - detected  : the image with drawn boxes (can be either the numpy array, either its filename, either None)
                    - result    : a `dict` with (at least) keys
                        - boxes     : list of boxes, the predicted positions of the objects
                        - timestamp : the timestamp at which the prediction has been performed
                        - filename (if `save` or filename)  : the original image filename
                        - detected (if `save_detected`)     : the filename of the image with boxes
                        - filename_boxes (if `save_boxes`)  : the filenames for the boxes images
            
            Note : videos are supported by this function but are simply forwarded to `self.predict_videos`. This distinction is important because the keys in `result` are different, and thus, to avoid any confusion, the mapping file is not the same (`map.json` vs `map_videos.json`).
            In this case, `kwargs` are forwarded to `self.predict_videos`
        """
        ####################
        # helping function #
        ####################
        
        time_logger.start_timer('initialization')

        @timer
        def post_process(idx):
            while idx < len(results) and results[idx] is not None:
                image, detected, infos = results[idx]
                
                if verbose and idx < display:
                    if isinstance(detected, str): detected = load_image(detected)
                    show_result_fn(image, detected = detected, boxes = infos.get('boxes', []))
                
                if post_processing is not None:
                    post_processing(detected, image = file, infos = infos)
                
                idx += 1

            return idx
        
        @timer
        def show_result(image, detected, boxes):
            if verbose > 1:
                if verbose == 3:
                    logger.info("{} boxes found :\n{}".format(
                        len(boxes), '\n'.join(str(b) for b in boxes)
                    ))

                show_boxes(image, boxes, labels = kwargs.get('labels', self.labels), ** plot_kwargs)
            # Show original image with drawed boxes
            plot(detected, title = '{} object(s) detected'.format(len(boxes)), ** plot_kwargs)

        def should_predict(image):
            if isinstance(image, (dict, pd.Series)) and 'filename' in image:
                image = image['filename']
            if isinstance(image, str) and all(k in predicted.get(image, {}) for k in required_keys):
                if not overwrite or (timestamp != -1 and timestamp <= predicted[image].get('timestamp', -1)):
                    return False
            return True
        
        def get_filename(image):
            if isinstance(image, (dict, pd.Series)):
                return image.get('filename', None)
            if isinstance(image, (np.ndarray, tf.Tensor)):
                return None
            elif isinstance(image, str):
                return image
            raise ValueError('Unknown image type ({}) : {}'.format(type(image), image))
        
        ####################
        #  Initialization  #
        ####################

        if saving_functions is None:
            saving_functions    = self._get_saving_functions(
                max_workers = max_workers, show_result_fn = show_result
            )

        if len(saving_functions) != 5:
            raise ValueError('`saving_functions` must be of length 5 !')
        
        show_result_fn, save_json_fn, save_image_fn, save_detected_fn, save_boxes_fn = saving_functions

        now = time.time()
        
        if isinstance(images, pd.DataFrame): images = images.to_dict('records')
        if not isinstance(images, (list, tuple, np.ndarray, tf.Tensor)): images = [images]
        elif isinstance(images, (np.ndarray, tf.Tensor)) and len(images.shape) == 3:
            images = np.expand_dims(images, axis = 0)

        if save_detected or save_boxes: save = True
        if (not save and post_processing is None) or display in (-1, True): display = len(images)
        if display: verbose = max(verbose, 1)
        
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        
        required_keys   = ['boxes']
        if save:
            if raw_img_dir is None: raw_img_dir = os.path.join(directory, 'images')
            if create_dirs: os.makedirs(raw_img_dir, exist_ok = True)
            required_keys.append('filename')
        
        if save_detected:
            if detected_dir is None: detected_dir = os.path.join(directory, 'detected')
            if create_dirs: os.makedirs(detected_dir, exist_ok = True)
            required_keys.append('detected')

        if save_boxes:
            if boxes_dir is None: boxes_dir = os.path.join(directory, 'boxes')
            if create_dirs: os.makedirs(boxes_dir, exist_ok = True)
            required_keys.append('filename_boxes')
        
        if predicted is None:
            predicted   = load_json(map_file, default = {})
        
        time_logger.stop_timer('initialization')

        ####################
        #  Pre-processing  #
        ####################
        
        time_logger.start_timer('pre-processing')
        
        results     = [None] * len(images)
        duplicatas  = {}
        requested   = [(get_filename(img), img) for img in images]
        
        videos, inputs, encoded = [], [], []
        for i, (file, img) in enumerate(requested):
            if not should_predict(file):
                results[i] = (file, predicted[file].get('detected', None), predicted[file])
            else:
                if isinstance(file, str):
                    duplicatas.setdefault(file, []).append(i)
                    
                    if file.endswith(_video_formats):
                        videos.append(file)
                        continue
                    elif len(duplicatas[file]) > 1:
                        continue
                
                if isinstance(img, dict):
                    if 'tf_image' in img:
                        image = img['tf_image']
                    elif 'image' in img:
                        image = load_image(img['image' if 'image_copy' not in img else 'image_copy'])
                    elif 'filename' in img:
                        image = load_image(img['filename'])
                else:
                    image = load_image(img)
                inputs.append((i, file, img, image))
                encoded.append(self.get_input(image))
        
        ####################
        #  Inference loop  #
        ####################
        
        show_idx    = post_process(0)
        
        if len(inputs) > 0:
            encoded = tf.stack(encoded, 0) if len(encoded) > 1 else tf.expand_dims(encoded[0], 0)
            encoded = self.preprocess_input(encoded)
            
            time_logger.stop_timer('pre-processing')
            
            for start in range(0, len(inputs), batch_size):
                # Computes detection + output decoding
                boxes   = self.detect(
                    encoded[start : start + batch_size], get_boxes = True, ** kwargs
                )
                
                should_save = False
                for (idx, file, data, image), box in zip(inputs[start : start + batch_size], boxes):
                    if file is None:
                        if isinstance(data, (np.ndarray, tf.Tensor)):
                            file = data
                        elif isinstance(data, (dict, pd.Series)) and 'image' in data:
                            file = data['image']
                        else:
                            file = image
                    # Maybe skips the image if nothing has been detected
                    if not isinstance(file, str) and not save_empty and len(box) == 0:
                        results[idx] = (file, file, {})
                        continue
                    
                    time_logger.start_timer('post processing')

                    detected    = None
                    basename    = None
                    if save_detected or verbose or post_processing is not None or force_draw:
                        time_logger.start_timer('drawing boxes')
                        
                        if isinstance(data, (dict, pd.Series)) and 'image_copy' in data:
                            detected = data['image_copy']
                        elif isinstance(file, np.ndarray):
                            detected = image.numpy() if save else file
                        elif isinstance(file, tf.Tensor):
                            detected = file.numpy()
                        else:
                            detected = image
                        
                        detected    = self.draw_prediction(detected, box, ** kwargs)
                        time_logger.stop_timer('drawing boxes')

                    infos   = {'boxes' : box, 'timestamp' : now}
                    if save:
                        time_logger.start_timer('saving frame')
                        # Saves the raw image (i.e. if it is not already a filename)
                        # The filename for the raw image is pre-computed here, even if `self.save_image` may do it, in order to allow multi-threading
                        if not isinstance(file, str):
                            if isinstance(data, (dict, pd.Series)) and 'frame_index' in data:
                                img_num = data['frame_index']
                            elif img_num == -1:
                                img_num = len(glob.glob(os.path.join(
                                    raw_img_dir, filename.replace('{}', '*')
                                )))
                            img_file = os.path.join(raw_img_dir, filename.format(img_num))
                            img_num += 1

                            save_image_fn(file, filename = img_file, directory = raw_img_dir)
                            file    = img_file
                        
                        basename    = os.path.splitext(os.path.basename(file))[0]
                        should_save     = True
                        predicted[file] = infos
                        time_logger.stop_timer('saving frame')

                    if isinstance(file, str):
                        infos['filename']   = file
                    
                    if save_detected:
                        time_logger.start_timer('saving detected')
                        detected_file = os.path.join(
                            detected_dir, detected_filename.format(basename)
                        )
                        save_detected_fn(
                            detected,
                            directory   = detected_dir,
                            filename    = detected_file
                        )
                        infos['detected'] = detected_file
                        time_logger.stop_timer('saving detected')

                    if save_boxes:
                        time_logger.start_timer('saving boxes')
                        box_files = [os.path.join(
                            boxes_dir, boxes_filename.format(basename, i)
                        ) for i in range(len(box))]
                        save_boxes_fn(
                            file,
                            image   = image,
                            boxes   = box,
                            labels  = self.labels,
                            directory   = boxes_dir,
                            file_format = box_files,
                            ** kwargs
                        )
                        infos['filename_boxes'] = {
                            box_file : get_box_infos(b, labels = self.labels, ** kwargs)
                            for box_file, b in zip(box_files, box)
                        }
                        time_logger.stop_timer('saving boxes')

                    # Sets result at the (multiple) index(es)
                    if isinstance(file, str) and file in duplicatas:
                        for duplicate_idx in duplicatas[file]:
                            results[duplicate_idx] = (file, detected, infos)
                    else:
                        results[idx] = (file, detected, infos)
                    
                    time_logger.stop_timer('post processing')
                
                if save and should_save:
                    time_logger.start_timer('saving json')
                    save_json_fn(map_file, data = predicted.copy(), indent = 4)
                    time_logger.stop_timer('saving json')
                
                show_idx = post_process(show_idx)
        else:
            time_logger.stop_timer('pre-processing')

        for fn in saving_functions:
            if isinstance(fn, Consumer): fn.join()

        if videos:
            kwargs.setdefault('save_frames', save)
            video_results = self.predict_video(
                videos,
                save    = save,
                save_detected   = save_detected,
                save_boxes      = save_boxes,
                
                directory   = directory,
                overwrite   = overwrite,
                max_workers = max_workers,
                
                ** kwargs
            )
            
            for file, infos in video_results:
                for duplicate_idx in duplicatas[file]:
                    results[duplicate_idx] = (file, infos.get('detected', None), infos)
        
        return results

    @timer
    def predict_video(self,
                      videos,
                      save  = True,
                      save_video = True,
                      save_frames   = False,
                      
                      directory = None,
                      overwrite = False,

                      tqdm  = lambda x: x,
                      
                      ** kwargs
                     ):
        """
            Perform prediction on `videos` (with `self.stream` method)
            
            Arguments :
                - videos    : (list of) video filenames
                - save      : whether to save the mapping file
                - save_video    : whether to save the result video with drawn boxes
                - save_frames   : whether to save each frame individually (see `save` in `self.predict`)
                
                - directory : where to save the mapping
                - overwrite : whether to overwrite (or not) the already predicted videos
                
                - tqdm  : progress bar
                
                - kwargs    : forwarded to `self.stream`
            Return :
                - list of tuple [(path, infos), ...]
                    - path  : original video path
                    - infos : general information on the prediction with keys
                        - width / height / fps / nb_frames  : general information on the video
                        - frames (if `save_frames`)     : filename for the frames mapping file (i.e. the output of `self.predict`)
                        - detected (if `save_video`)    : the path to the output video
        """
        if not save_frames: kwargs.update({'save_detected' : False, 'save_boxes' : False})
        kwargs.setdefault('show', False)
        kwargs.setdefault('max_time', -1)
        
        videos = normalize_filename(videos)
        if not isinstance(videos, (list, tuple)): videos = [videos]
        
        # get saving directory
        if directory is None: directory = self.pred_dir
        
        map_file    = os.path.join(directory, 'map_videos.json')
        infos_videos    = load_json(map_file, default = {})
        
        video_dir = os.path.join(directory, 'videos')
        
        # Filters files that do not end with a valid video extension
        videos = [video for video in videos if video.endswith(_video_formats)]
        
        for path in tqdm(set(videos)):
            video_name, ext  = os.path.splitext(os.path.basename(path))
            # Maybe skip because already predicted
            if not overwrite and path in infos_videos:
                if not save_frames or (save_frames and infos_videos[path]['frames'] is not None):
                    if not save_video or (save_video and infos_videos[path]['detected'] is not None):
                        continue
            
            save_dir    = os.path.join(video_dir, video_name)
            out_file = None if not save_video else os.path.join(
                save_dir, '{}_detected{}'.format(video_name, ext)
            )
            map_frames = os.path.join(save_dir, 'map.json') if save_frames else None
            
            if os.path.exists(save_dir): shutil.rmtree(save_dir)
            if out_file and os.path.exists(out_file): os.remove(out_file)
            if out_file: os.makedirs(save_dir, exist_ok = True)
            
            self.stream(
                cam_id  = path,
                save    = save_frames,
                directory   = save_dir,
                output_file = out_file,
                
                ** kwargs
            )
            
            infos   = get_video_infos(path).__dict__
            if out_file:    infos['detected'] = out_file
            if save_frames: infos['frames'] = os.path.join(save_dir, 'map.json')

            infos_videos[path] = infos
        
            dump_json(map_file, infos_videos, indent = 4)
        
        return [(video, infos_videos[video]) for video in videos]

    def evaluate(self, 
                 generator, 
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """    
        return -1.
        # gather all detections and annotations
        generator.batch_size = 1
        all_detections     = [[None for i in range(len(self.labels))] for j in range(len(generator))]
        all_annotations    = [[None for i in range(len(self.labels))] for j in range(len(generator))]

        for i in range(len(generator)):
            inputs, _ = generator.__getitem__(i)
            raw_image, _ = inputs
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self._predict(inputs, get_boxes=True)
            
            score = np.array([box.get_score() for box in pred_boxes])
            pred_labels = np.array([box.get_label() for box in pred_boxes])        
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(len(self.labels)):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        return average_precisions
                                
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            'anchors'   : self.anchors,
            'backend'   : self.backend,
            
            'labels'    : self.labels,
            'nb_class'  : self.nb_class,
            'max_box_per_image' : self.max_box_per_image
        })
        
        return config

    @classmethod
    def from_darknet_pretrained(cls,
                                weight_path = 'yolov2.weights',
                                nom     = 'coco_pretrained',
                                labels  = COCO_CONFIG['labels'],
                                ** kwargs
                               ):
        if not os.path.exists(weight_path) and weight_path.endswith('yolov2.weights'):
            weight_path = download_file(PRETRAINED_COCO_URL, filename = weight_path)
            
        instance = cls(
            nom = nom, labels = labels, max_to_keep = 1, pretrained_name = weight_path, ** kwargs
        )
        
        decode_darknet_weights(instance.get_model(), weight_path)
        
        instance.save()
        
        return instance

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_darknet_weights(model, wt_path):
    #Chargement des poids
    weight_reader = WeightReader(wt_path)
    weight_reader.reset()
    nb_conv = 23
    for i in range(1, nb_conv+1):
        name = 'conv_{}'.format(i) if i < 23 else 'detection_layer'
        if i < nb_conv and len(model.layers) < 10:
            conv_layer = model.layers[1].get_layer(name = name)
            norm_layer = model.layers[1].get_layer(
                'batch_normalization_{}'.format(i-1) if i > 1 else 'batch_normalization'
            )
        else:
            conv_layer = model.get_layer(name = name)
            norm_layer = model.get_layer(
                'batch_normalization_{}'.format(i-1) if i > 1 else 'batch_normalization'
            ) if i < nb_conv else None
        
        if (i < nb_conv):
            size = np.prod(norm_layer.get_weights()[0].shape)
            
            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])
        
        if (len(conv_layer.get_weights()) > 1):
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])

class WeightReader:
    def __init__(self, path):
        self.offset = 4
        self.all_weights = np.fromfile(path, dtype = 'float32')
    
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size : self.offset]
    
    def reset(self):
        self.offset = 4
