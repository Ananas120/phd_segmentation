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
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

from utils.generic_utils import convert_to_str
from utils.med_utils.label_utils import rearrange_labels, build_mapping, transform_mask
from utils.med_utils.resampling import resample_volume
from utils.med_utils.pre_processing import extract_slices

""" Utility functions """

def get_nb_frames(data, axis = 2):
    if isinstance(data, (dict, pd.Series)):
        if 'nb_images' in data: return data['nb_images']
        if 'nb_frames' in data: return data['nb_frames']
        data = data['images']
    
    if not isinstance(data, (list, str)): raise ValueError('Invalid data (type {}) : {}'.format(type(data), data))
    
    return get_shape(data)[axis]

def get_shape(filename):
    if filename.endswith(('.nii.gz', '.nii')):
        import nibabel as nib
        return nib.load(filename).shape
    elif filename.endswith('.npz'):
        with np.load(filename) as file:
            return file['shape']
    else:
        return _get_shape_from_filename(filename)

def get_voxel_dims(filename):
    if filename.endswith(('.nii.gz', '.nii')):
        import nibabel as nib
        return nib.load(filename).header['pixdim'][1:4]
    elif filename.endswith('.npz'):
        with np.load(filename) as file:
            voxel_dims = file['pixdim']
            if len(voxel_dims) > 3: voxel_dims = voxel_dims[1:4]
            return voxel_dims
    else:
        return _get_pixdims_from_filename(filename)
    
@tf.function(experimental_follow_type_hints = True)
def _tf_get_shape_from_filename(filename : tf.Tensor, ext_length : tf.Tensor, dtype = tf.int32):
    filename = tf.strings.substr(filename, 0, tf.strings.length(filename) - ext_length)
    return tf.strings.to_number(tf.strings.split(filename, '-')[4:], out_type = dtype)

def _get_shape_from_filename(filename):
    basename = '.'.join(os.path.basename(filename).split('.')[:-1])
    return tuple(int(v) for v in basename.split('-')[4:])


@tf.function(experimental_follow_type_hints = True)
def _tf_get_pixdims_from_filename(filename : tf.Tensor, ext_length : tf.Tensor):
    filename = tf.strings.substr(filename, 0, tf.strings.length(filename) - ext_length)
    return tf.strings.to_number(tf.strings.split(filename, '-')[1:4])

@tf.function(input_signature = [tf.TensorSpec(shape = (None, ), dtype = tf.string)])
def _tf_get_pixdims_from_series(filenames : tf.Tensor):
    f0 = tf.io.read_file(filenames[0])
    
    hw_spacing = tf.strings.to_number(tf.strings.split(
        tfio.image.decode_dicom_data(f0, tfio.image.dicom_tags.PixelSpacing), b'\\'
    ))
    hw_spacing.set_shape([2])
    
    if len(filenames) == 1: return tf.concat([hw_spacing, tf.cast([-1.], hw_spacing.dtype)], axis = -1)

    f1 = tf.io.read_file(filenames[1])
    z_pos1 = tf.strings.to_number(tf.strings.split(
        tfio.image.decode_dicom_data(f0, tfio.image.dicom_tags.ImagePositionPatient), b'\\'
    ))[-1:]
    z_pos2 = tf.strings.to_number(tf.strings.split(
        tfio.image.decode_dicom_data(f1, tfio.image.dicom_tags.ImagePositionPatient), b'\\'
    ))[-1:]

    return tf.concat([hw_spacing, z_pos2 - z_pos1], axis = -1)

def _get_pixdims_from_filename(filename):
    return tuple(float(v) for v in os.path.basename(filename)[:-4].split('-')[1:4])

def format_filename(filename, shape, voxel_dims):
    basename, ext = os.path.splitext(filename) if not filename.endswith('.nii.gz') else (filename[:-7], '.nii.gz')
    if isinstance(shape, tf.Tensor): shape = shape.numpy()
    str_shape = '-'.join([str(s) for s in tuple(shape)])
    str_voxel = '-'.join(['{:.1f}'.format(v) for v in tuple(voxel_dims)])
    return '{}-{}-{}{}'.format(basename, str_voxel, str_shape, ext)

""" Loading functions """

def load_medical_data(filename,
                      
                      slice_start = -1,
                      slice_end   = -1,
                      slice_axis  = 2,
                      
                      labels      = None,
                      mapping     = None,
                      
                      use_sparse  = False,
                      is_one_hot  = True,
                      dtype       = tf.float32,
                      
                      ** kwargs
                     ):
    def _eager_load(filename, start, end, axis, labels = None, mapping = None):
        return load_medical_data(
            convert_to_str(filename),
            slice_start = start,
            slice_end   = end,
            slice_axis  = axis,
            labels      = labels,
            mapping     = mapping,
            use_sparse  = use_sparse,
            dtype       = dtype,
            ** kwargs
        )
    
    if labels is not None and mapping is not None:
        _label_args = [tf.cast(labels, tf.string), build_mapping(mapping, output_format = 'tensor')]
    else:
        _label_args = []
    
    if isinstance(filename, list): filename = tf.cast(filename, tf.string)
    
    post_process = True
    data_type    = dtype if not use_sparse else tf.SparseTensorSpec(shape = None, dtype = dtype)
    if isinstance(filename, tf.Tensor) and filename.dtype == tf.string:
        if not use_sparse and len(tf.shape(filename)) == 1:
            post_process     = False
            data, voxel_dims = _tf_load_dicom_series(
                filename, start = slice_start, end = slice_end, ** kwargs
            )
        elif use_sparse and tf.strings.regex_full_match(filename, '.*stensor$'):
            data, voxel_dims = _tf_load_sparse(filename, dtype = dtype, ** kwargs)
        elif not use_sparse and tf.strings.regex_full_match(filename, '.*tensor$'):
            data, voxel_dims = _tf_load(filename, dtype = dtype, ** kwargs)
        else:
            post_process     = False
            data, voxel_dims = tf.py_function(
                _eager_load, [filename, slice_start, slice_end, slice_axis] + _label_args, Tout = [data_type, tf.float32]
            )
    elif isinstance(filename, str):
        ext = max(_loading_fn.keys(), key = lambda ext: -1 if not filename.endswith(ext) else len(ext))

        if not filename.endswith(ext):
            raise ValueError('Unsupported file type : {}'.format(os.path.basename(filename)))
        
        data, voxel_dims = _loading_fn[ext](filename, dtype = dtype, ** kwargs)
    elif isinstance(filename, dict):
        post_process = False
        if not tf.executing_eagerly():
            data, voxel_dims = tf.py_function(
                _eager_load, [filename, slice_start, slice_end, slice_axis] + _label_args, Tout = [data_type, tf.float32]
            )
        else:
            val_to_idx = build_mapping(mapping, output_format = 'dict')

            data = None
            for i, (label, file) in enumerate(convert_to_str(filename).items()):
                if val_to_idx and label not in val_to_idx: continue

                data_i, voxel_dims = load_medical_data(
                    file, slice_start = slice_start, slice_end = slice_end, slice_axis = slice_axis, dtype = dtype, ** kwargs
                )
                if data is None: data = np.zeros(data_i.shape, dtype = np.int32)
                data[data_i.astype(bool)] = val_to_idx.get(label, i + 1)
    else:
        data, voxel_dims = filename
    
    if use_sparse and isinstance(data, tf.sparse.SparseTensor):
        data.indices.set_shape([None, None])
        data.values.set_shape([None])
        data.dense_shape.set_shape([None])
    
    if post_process:
        if slice_start != -1 or slice_end != -1:
            data = extract_slices(data, start = slice_start, end = slice_end, axis = slice_axis)

        if not isinstance(filename, dict) and labels is not None and mapping is not None:
            data = rearrange_labels(
                data, labels, mapping = mapping, is_one_hot = is_one_hot, default = 0
            )

    if use_sparse and not isinstance(data, tf.sparse.SparseTensor):
        data = tf.sparse.from_dense(data)
    
    if isinstance(data, np.ndarray):
        if dtype == tf.uint8:     np_dtype = np.uint8
        elif dtype == tf.bool:    np_dtype = bool
        elif dtype == tf.int32:   np_dtype = np.int32
        elif dtype == tf.float32: np_dtype = np.float32
        else: np_dtype = dtype
        data = data.astype(np_dtype)
    else:
        data = tf.cast(data, dtype)
    return data, tf.cast(voxel_dims, tf.float32)

def _pydicom_load(filename, return_label = False, ** kwargs):
    import pydicom as dcm
    
    file = dcm.dcmread(filename)
    return file.pixel_array.astype(np.int16), np.array(list(file.PixelSpacing) + [-1], dtype = np.float32)

def _nibabel_load(filename, return_label = False, ** kwargs):
    import nibabel as nib
    
    file    = nib.load(filename)
    data    = file.get_fdata(caching = 'unchanged')
    pixdims = file.header['pixdim'][1:4]
    labels  = None
    if return_label:
        if not file.extra or 'labels' not in file.extra:
            raise RuntimeError('The `labels` have not been saved in the Nifti file !')
        labels = file.extra['labels']
    
    return (data, pixdims) if not return_label else (data, pixdims, labels)

def _numpy_loadz(filename, return_label = False, ** kwargs):
    with np.load(filename) as file:
        voxel_dims = file['pixdim']
        if len(voxel_dims) > 3: voxel_dims = voxel_dims[1:4]
        labels     = file['labels'] if return_label and 'labels' in file else None
        if 'data' in file:
            data = file['data']
        elif 'mask' in file or 'indices' in file:
            data = file['mask'] if 'mask' in file else file['indices']
            if 'shape' in file:
                data = tf.sparse.SparseTensor(
                    indices     = data,
                    values      = tf.ones((len(data), ), dtype = tf.uint8),
                    dense_shape = file['shape']
                )
    
    if return_label and not labels: raise RuntimeError('The `labels` was not saved in the `.npz` file !')
    
    return (data, voxel_dims) if not return_label else (data, voxel_dims, labels)

def _numpy_load(filename, return_label = False, ** kwargs):
    if return_label: raise RuntimeError('The `npy` format does not contain label information')
    
    data     = np.load(filename)
    vox_dims = _get_pixdims_from_filename(filename)
    return data, vox_dims

def _tf_load(filename, shape = (None, None, None), dtype = tf.float32, ext_length = 7, return_label = False, ** kwargs):
    if return_label: raise RuntimeError('The `tensor` format does not contain label information')
    
    data = tf.ensure_shape(tf.io.parse_tensor(tf.io.read_file(filename), dtype), shape)
    return (data, _tf_get_pixdims_from_filename(filename, ext_length))

def _tf_load_sparse(filename, dtype = tf.uint8, return_label = False, ** kwargs):
    if return_label: raise RuntimeError('The `stensor` format does not contain label information')
    
    if isinstance(dtype, tf.SparseTensorSpec): dtype = dtype.dtype
    
    indices, voxel_dims = _tf_load(filename, shape = (None, None), dtype = tf.int64, ext_length = 8)
    shape = _tf_get_shape_from_filename(filename, 8, dtype = tf.int64)
    shape.set_shape([indices.shape[1]])
    return tf.sparse.SparseTensor(
        indices     = indices,
        values      = tf.ones((tf.shape(indices)[0], ), dtype = dtype),
        dense_shape = shape
    ), voxel_dims

@tf.function(experimental_follow_type_hints = True)
def _tf_load_dicom_series(filename : tf.Tensor,
                          start    : tf.Tensor = -1,
                          end      : tf.Tensor = -1,
                          dtype    = tf.float32,
                          
                          use_pydicom = True,
                          ** kwargs
                         ):
    if start == -1: start = 0
    if end == -1:   end = tf.shape(filename)[0]
    
    volume = tf.TensorArray(dtype = dtype, size = end - start)
    for i in tf.range(start, end):
        if use_pydicom:
            frame, _ = tf.numpy_function(
                _pydicom_load, [filename[i]], Tout = [tf.int16, tf.float32]
            )
            frame.set_shape([None, None])
        else:
            file   = tf.io.read_file(filename[i])
            frame  = tfio.image.decode_dicom_image(file, on_error = 'lossy', scale = 'auto', dtype = tf.uint16)
        frame  = tf.image.convert_image_dtype(frame[0, :, :, 0], dtype)
        volume = volume.write(i - start, frame)
    
    return tf.transpose(volume.stack(), [1, 2, 0]), _tf_get_pixdims_from_series(filename)

""" Saving functions """

def save_medical_data(filename, data, voxel_dims, ** kwargs):
    ext = max(_saving_fn.keys(), key = lambda ext: -1 if not filename.endswith(ext) else len(ext))

    if not filename.endswith(ext):
        raise ValueError('Unsupported file type : {}'.format(os.path.basename(filename)))
    
    return _saving_fn[ext](filename, data, voxel_dims = voxel_dims, ** kwargs)

def _nibabel_save(filename, data, affine, header, ** kwargs):
    import nibabel as nib
    
    nifti = nib.Nifti1Image(data, affine, header = header, extra = kwargs)
    return nib.save(nifti, filename)

def _numpy_savez(filename, data, voxel_dims, affine = None, labels = None, compressed = True, ** kwargs):
    additionals = {}
    if affine is not None: additionals['affine'] = affine
    if labels is not None: additionals['labels'] = labels
    if isinstance(data, tf.sparse.SparseTensor):
        indices = data.indices.numpy()
        shape   = data.dense_shape
    else:
        indices = np.stack(np.where(data), axis = -1)
        shape   = data.shape
    
    saving_fn = np.savez if not compressed else np.savez_compressed
    saving_fn(filename, indices = indices, pixdim = np.array(voxel_dims), shape = np.array(shape), ** additionals)
    return filename

def _numpy_save(filename, data, voxel_dims, ** kwargs):
    filename = format_filename(filename, shape = data.shape, voxel_dims = voxel_dims)
    np.save(filename, data)
    return filename

def _tf_save(filename, data, voxel_dims, ** kwargs):
    filename = format_filename(filename, shape = data.shape, voxel_dims = voxel_dims)
    tf.io.write_file(
        filename, tf.io.serialize_tensor(tf.cast(data, tf.float32))
    )
    return filename

def _tf_save_sparse(filename, data, voxel_dims, ** kwargs):
    if not isinstance(data, tf.sparse.SparseTensor): data = tf.sparse.from_dense(data)
    filename = format_filename(filename, shape = data.dense_shape, voxel_dims = voxel_dims)
    tf.io.write_file(
        filename, tf.io.serialize_tensor(data.indices)
    )
    return filename

_loading_fn = {
    'nii'     : _nibabel_load,
    'nii.gz'  : _nibabel_load,
    'npy'     : _numpy_load,
    'npz'     : _numpy_loadz,
    'dcm'     : _pydicom_load,
    'tensor'  : _tf_load,
    'stensor' : _tf_load_sparse
}

_saving_fn = {
    'nii'     : _nibabel_save,
    'nii.gz'  : _nibabel_save,
    'npy'     : _numpy_save,
    'npz'     : _numpy_savez,
    'tensor'  : _tf_save,
    'stensor' : _tf_save_sparse
}