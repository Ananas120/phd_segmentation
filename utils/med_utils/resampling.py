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

import numpy as np
import tensorflow as tf

from utils.image import resize_image

def _expand(img_shape, dims):
    if not isinstance(dims, (list, tuple, np.ndarray, tf.Tensor)):
        dims = tf.fill((tf.minimum(len(img_shape), 3), ), dims)
    return dims

def _multiply(shape, factors):
    n_dims_to_resize    = tf.shape(factors)[0]
    
    return tf.maximum(1, tf.concat([
        tf.cast(tf.math.round(tf.cast(shape[: n_dims_to_resize], tf.float32) * factors), tf.int32),
        shape[n_dims_to_resize :]
    ], axis = -1))

def compute_new_shape(img_shape, voxel_dims, target_voxel_dims, return_factors = False):
    """
        Compute the new shape of a volume given a voxel dims to a new voxel dims
        
        Arguments :
            - img_shape  : the volume / image shape
            - voxel_dims : the initial voxel dimensions, i.e. the size (in mm) of a single voxel in the 3-D space of the volume (or 2-D space of the image)
            - target_voxel_dims : the new dimension of voxels (in mm)
        Return :
            - new_shape : tf.Tensor the same length as img_shape with the 3 first dimensions (possibly) modified
    """
    voxel_dims          = tf.cast(_expand(img_shape, voxel_dims), tf.float32)
    target_voxel_dims   = tf.cast(_expand(img_shape, target_voxel_dims), tf.float32)
    
    factors   = voxel_dims / target_voxel_dims
    if return_factors: return factors
    return _multiply(img_shape, factors)

def compute_new_voxel_dims(img_shape, voxel_dims, target_shape):
    """
        Compute the new shape of a volume given a voxel dims to a new voxel dims
        
        Arguments :
            - img_shape  : the volume / image shape
            - voxel_dims : the initial voxel dimensions, i.e. the size (in mm) of a single voxel in the 3-D space of the volume
            - target_shape : the new shape for the image / volume
        Return :
            - new_voxel_dims : tf.Tensor with same length as voxel_dims (if list / tuple) or min(len(img_shape), 3) (i.e. if `img_shape` is 4-D, only the 3 first dimensions are used)
    """
    voxel_dims  = tf.cast(_expand(img_shape, voxel_dims), tf.float32)
    n_dims_to_resize    = tf.shape(voxel_dims)[0]
    
    img_shape   = tf.cast(img_shape[: n_dims_to_resize], tf.float32)
    target_shape    = tf.cast(target_shape[: n_dims_to_resize], tf.float32)

    return voxel_dims * (img_shape / target_shape)

def resample_volume(volume,
                    voxel_dims,
                    method      = 'tensorflow',
                    target_shape    = None,
                    target_voxel_dims   = None,
                    ** kwargs
                   ):
    """
        Resizes a 3-D or 4-D volume to a new shape, either specified as a strict new shape, either as a new voxel dimensions.
        In the 2nd case, the shape is dynamically computed based on the original voxel dimension
        
        Arguments :
            - img        : the 3-D or 4-D volume
            - voxel_dims : the initial voxel dimensions, i.e. the size (in mm) of a single voxel in the 3-D space of the volume
            - method     : the resampling method
            
            - target_voxel_dims : the expected new size of voxels (in mm) of a single voxel in the 3-D space of the volume
            - target_shape      : the new shape for the volume
            - kwargs            : forwarded to the resampling method
        Return :
            - resized_img    : np.ndarray or tf.Tensor with the same number of dimensions as `img` with the 3 first dimensions possibly resized
            - new_voxel_dims : the new voxel dimensions of the resized volume
    """
    assert target_voxel_dims is not None or target_shape is not None
    
    if method not in _resampling_methods:
        raise ValueError('Unknown resampling method !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_resampling_methods.keys()), method
        ))
        
    if target_shape is None:
        target_shape = compute_new_shape(tf.shape(volume), voxel_dims, target_voxel_dims)
    if target_voxel_dims is None:
        target_voxel_dims = compute_new_voxel_dims(tf.shape(volume), voxel_dims, target_shape)
    
    return _resampling_methods[method](
        volume,
        voxel_dims      = voxel_dims,
        target_shape    = target_shape,
        target_voxel_dims   = target_voxel_dims,
        ** kwargs
    ), target_voxel_dims

def _resample_tensorflow(volume,
                         target_shape   = None,
                         interpolation  = 'bilinear',
                         preserve_aspect_ratio  = True,
                         ** kwargs
                        ):
    """ `volume` is a `tf.Tensor` of 2D (image), 3D (volume) or 4D (volume + temporal) """
    if not isinstance(volume, (tf.Tensor, tf.sparse.SparseTensor)):
        volume = tf.cast(volume, tf.float32)

    is_sparse = isinstance(volume, tf.sparse.SparseTensor)
    dim = len(tf.shape(volume))
    
    resized = volume
    if tf.reduce_any(tf.shape(volume)[: len(target_shape)] != target_shape):
        if is_sparse: resized = tf.sparse.to_dense(resized)
        if dim == 2:  resized = tf.expand_dims(resized, axis = -1)
        if dim <= 3:  resized = tf.expand_dims(resized, axis = -1)
        
        if tf.shape(volume)[0] != target_shape[0] or tf.shape(volume)[1] != target_shape[1]:
            resized = tf.transpose(resize_image(
                tf.transpose(resized, [2, 0, 1, 3]),
                target_shape[:2],
                method = interpolation,
                preserve_aspect_ratio = preserve_aspect_ratio
            ), [1, 2, 0, 3])

        if dim > 2 and tf.shape(volume)[2] != target_shape[2]:
            resized = resize_image(
                resized,
                target_shape[1:3],
                method = interpolation,
                preserve_aspect_ratio = preserve_aspect_ratio
            )
        
        if dim == 2:    resized = resized[:, :, 0, 0]
        if dim == 3:    resized = resized[:, :, :, 0]
        if is_sparse:   resized = tf.sparse.from_dense(resized)

    return resized

def _resample_nilearn(img, voxel_dims, target_shape = None, mode = 'continuous'):
    import nibabel as nib
    import nilearn.image as niimg
    
    if isinstance(img, str): img = nib.load(img)
    
    rescaled_affine = rescale_affine(img, voxel_dims, target_shape = target_shape)
    
    return niimg.resample_img(img, rescaled_affine, target_shape = target_shape, interpolation = mode).get_fdata()

def rescale_affine(img, voxel_dims = (1, 1, 1), target_shape = None, target_center_coords = None):
    """
        Comes from https://github.com/nipy/nibabel/issues/670

        This function uses a generic approach to rescaling an affine to arbitrary
        voxel dimensions. It allows for affines with off-diagonal elements by
        decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
        and applying the scaling to the scaling matrix (s).

        Parameters
        ----------
        input_affine : np.array of shape 4,4
            Result of nibabel.nifti1.Nifti1Image.affine
        voxel_dims : list
            Length in mm for x,y, and z dimensions of each voxel.
        target_center_coords: list of float
            3 numbers to specify the translation part of the affine if not using the same as the input_affine.

        Returns
        -------
        target_affine : 4x4matrix
            The resampled image.
    """
    if not isinstance(voxel_dims, (tuple, list, np.ndarray)): voxel_dims = [voxel_dims] * 3
    if isinstance(img, str): img = nib.load(img)
    # Initialize target_affine
    target_affine = img.affine.copy()
    # Decompose the image affine to allow scaling
    u, s, v = np.linalg.svd(target_affine[:3, :3], full_matrices = False)
    
    # Rescale the image to the appropriate voxel dimensions
    s = np.array(voxel_dims)
    
    # Reconstruct the affine
    target_affine[:3, :3] = u @ np.diag(s) @ v
    
    if target_shape and target_center_coords is None:
        # Calculate the translation part of the affine
        spatial_dimensions = (img.header['dim'] * img.header['pixdim'])[1:4]

        # Calculate the translation affine as a proportion of the real world
        # spatial dimensions
        image_center_as_prop = img.affine[0:3, 3] / spatial_dimensions

        # Calculate the equivalent center coordinates in the target image
        dimensions_of_target_image = (np.array(voxel_dims) * np.array(target_shape))
        target_center_coords =  dimensions_of_target_image * image_center_as_prop 

    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3,3] = target_center_coords
    return target_affine

_resampling_methods = {
    'tensorflow' : _resample_tensorflow,
    'nilearn'    : _resample_nilearn
}