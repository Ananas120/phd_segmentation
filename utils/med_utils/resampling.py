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
import nibabel as nib
import tensorflow as tf

from utils.image import resize_image

def compute_new_shape(img_shape, voxel_dims, target_voxel_dims):
    """
        Compute the new shape of a volume given a voxel dims to a new voxel dims
        
        Arguments :
            - img_shape  : either the volume itself (np.ndarray) either its shape
            - voxel_dims : the initial voxel dimensions, i.e. the size (in mm) of a single voxel in the 3-D space of the volume
            - target_voxel_dims : the new dimension of voxels (in mm)
        Return :
            - new_shape : tuple of the same length as img_shape with the 3 first dimensions (possibly) modified
        
        Note : img_shape can be a 4-D shape but only the 3 first dimensions are modified, as the 4th one is typically the time
    """
    if not isinstance(voxel_dims, (list, tuple, np.ndarray, tf.Tensor)): voxel_dims = [voxel_dims] * 3
    if not isinstance(target_voxel_dims, (list, tuple, np.ndarray, tf.Tensor)): target_voxel_dims = [target_voxel_dims] * 3
    
    factors_xyz = tf.cast(voxel_dims, tf.float32) / tf.cast(target_voxel_dims, tf.float32)
    return tf.maximum(1, tf.concat([
        tf.cast(tf.cast(img_shape[:3], tf.float32) * factors_xyz, tf.int32),
        img_shape[3:]
    ], axis = -1))

def compute_new_voxel_dims(img_shape, voxel_dims, target_shape):
    """
        Compute the new shape of a volume given a voxel dims to a new voxel dims
        
        Arguments :
            - img_shape  : either the volume itself (np.ndarray) either its shape
            - voxel_dims : the initial voxel dimensions, i.e. the size (in mm) of a single voxel in the 3-D space of the volume
            - target_shape : the new shape for the volume
        Return :
            - new_shape : tuple of the same length as img_shape with the 3 first dimensions (possibly) modified
        
        Note : img_shape can be a 4-D shape but only the 3 first dimensions are modified, as the 4th one is typically the time
    """
    if not isinstance(voxel_dims, (list, tuple, np.ndarray, tf.Tensor)): voxel_dims = [voxel_dims] * 3
    
    factors = tf.cast(img_shape[:3], tf.float32) / tf.cast(target_shape[:3], tf.float32)
    return tf.concat([
        tf.cast(tf.cast(voxel_dims, tf.float32) / factors, tf.int32),
        img_shape[3:]
    ], axis = -1)

def resample_volume(img, voxel_dims, method = 'tensorflow', target_voxel_dims = None, target_shape = None, ** kwargs):
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
        
    if target_shape is None:      target_shape = compute_new_shape(tf.shape(img), voxel_dims, target_voxel_dims)
    if target_voxel_dims is None: target_voxel_dims = compute_new_voxel_dims(tf.shape(img), voxel_dims, target_shape)
    
    return _resampling_methods[method](
        img, voxel_dims = voxel_dims, target_voxel_dims = target_voxel_dims, target_shape = target_shape, ** kwargs
    ), target_voxel_dims

def _resample_tensorflow(img, target_shape = None, preserve_aspect_ratio = True, interpolation = 'bilinear', ** kwargs):
    """ The image must be at least 3D tensor of shape [height, width, depth, (optioinal) n_channels] """
    if not isinstance(img, (tf.Tensor, tf.sparse.SparseTensor)): img = tf.cast(img, tf.float32)

    is_sparse = isinstance(img, tf.sparse.SparseTensor)
    dim = len(tf.shape(img))
    
    resized = img
    if tf.shape(img)[0] != target_shape[0] or tf.shape(img)[1] != target_shape[1] or tf.shape(img)[2] != target_shape[2]:
        if is_sparse: resized = tf.sparse.to_dense(resized)
        if dim == 3:  resized = tf.expand_dims(resized, axis = -1)
        
        if tf.shape(img)[0] != target_shape[0] or tf.shape(img)[1] != target_shape[1]:
            resized = tf.transpose(resize_image(
                tf.transpose(resized, [2, 0, 1, 3]), target_shape[:2], method = interpolation, preserve_aspect_ratio = preserve_aspect_ratio
            ), [1, 2, 0, 3])

        if tf.shape(img)[2] != target_shape[2]:
            resized = resize_image(
                resized, target_shape[1:3], method = interpolation, preserve_aspect_ratio = preserve_aspect_ratio
            )
        
        if dim == 3:  resized = resized[..., 0]
        if is_sparse: resized = tf.sparse.from_dense(resized)

    return resized

def _resample_nilearn(img, voxel_dims, target_shape = None, mode = 'continuous'):
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