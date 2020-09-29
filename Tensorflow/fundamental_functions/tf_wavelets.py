"""
Wavelet transforms.
Tensorflow version.

Codes widely inspired from :
    K. Haug and M. Lohne, TF-Wavelets, 2019. Link: https://github.com/UiO-CS/tfwavelets.
Modified and enhanced by Mathilde Galinier
    
@date: 29/09/2020    
"""

import numpy as np
import tensorflow as tf

def adapt_filter(filter):
    """
    Expands dimensions of a 1d vector to match the required tensor dimensions in a TF
    graph.

    Args:
        filter (np.ndarray):     A 1D vector containing filter coefficients

    Returns:
        np.ndarray: A 3D vector with two empty dimensions as dim 2 and 3.

    """
    # Add empty dimensions for batch size and channel num
    return np.expand_dims(np.expand_dims(filter, -1), -1)

def to_tf_mat(matrices,tf_prec):
    """
    Expands dimensions of 2D matrices to match the required tensor dimensions in a TF
    graph, and wrapping them as TF constants.

    Args:
        matrices (iterable):    A list (or tuple) of 2D numpy arrays.

    Returns:
        iterable: A list of all the matrices converted to 3D TF tensors.
    """
    result = []

    for matrix in matrices:
        result.append(
            tf.constant(np.expand_dims(matrix, 0), dtype=tf_prec)
        )
    return result

class Filter:
    """
    Class representing a filter.

    Attributes:
        coeffs (tf.constant):      Filter coefficients
        zero (int):                Origin of filter (which index of coeffs array is
                                   actually indexed as 0).
        edge_matrices (iterable):  List of edge matrices, used for circular convolution.
                                   Stored as 3D TF tensors (constants).
    """

    def __init__(self, coeffs, zero,tf_prec,np_prec):
        """
        Create a filter based on given filter coefficients

        Args:
            coeffs (np.ndarray):       Filter coefficients
            zero (int):                Origin of filter (which index of coeffs array is
                                       actually indexed as 0).
            tf_prec:      tf.float16 ou tf.float32 ou tf.float64
        """
        self.coeffs = tf.constant(adapt_filter(coeffs), dtype=tf_prec)

        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(self.coeffs)
        self._coeffs = coeffs.astype(np_prec)

        self.zero = zero

        self.edge_matrices = to_tf_mat(self._edge_matrices(),tf_prec)

    def __getitem__(self, item):
        """
        Returns filter coefficients at requested indeces. Indeces are offset by the filter
        origin

        Args:
            item (int or slice):    Item(s) to get

        Returns:
            np.ndarray: Item(s) at specified place(s)
        """
        if isinstance(item, slice):
            return self._coeffs.__getitem__(
                slice(item.start + self.zero, item.stop + self.zero, item.step)
            )
        else:
            return self._coeffs.__getitem__(item + self.zero)

    def num_pos(self):
        """
        Number of positive indexed coefficients in filter, including the origin. Ie,
        strictly speaking it's the number of non-negative indexed coefficients.

        Returns:
            int: Number of positive indexed coefficients in filter.
        """
        return len(self._coeffs) - self.zero

    def num_neg(self):
        """
        Number of negative indexed coefficients, excluding the origin.

        Returns:
            int: Number of negative indexed coefficients
        """
        return self.zero

    def _edge_matrices(self):
        """Computes the submatrices needed at the ends for circular convolution.

        Returns:
            Tuple of 2d-arrays, (top-left, top-right, bottom-left, bottom-right).
        """
        if not isinstance(self._coeffs, np.ndarray):
            self._coeffs = np.array(self._coeffs)

        n, = self._coeffs.shape
        self._coeffs = self._coeffs[::-1]

        # Some padding is necesssary to keep the submatrices
        # from having having columns in common
        padding = max((self.zero, n - self.zero - 1))
        matrix_size = n + padding
        filter_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        negative = self._coeffs[
                   -(self.zero + 1):]  # negative indexed filter coeffs (and 0)
        positive = self._coeffs[
                   :-(self.zero + 1)]  # filter coeffs with strictly positive indeces

        # Insert first row
        filter_matrix[0, :len(negative)] = negative

        # Because -0 == 0, a length of 0 makes it impossible to broadcast
        # (nor is is necessary)
        if len(positive) > 0:
            filter_matrix[0, -len(positive):] = positive

        # Cycle previous row to compute the entire filter matrix
        for i in range(1, matrix_size):
            filter_matrix[i, :] = np.roll(filter_matrix[i - 1, :], 1)

        # TODO: Indexing not thoroughly tested
        num_pos = len(positive)
        num_neg = len(negative)
        top_left = filter_matrix[:num_pos, :(num_pos + num_neg - 1)]
        top_right = filter_matrix[:num_pos, -num_pos:]
        bottom_left = filter_matrix[-num_neg + 1:, :num_neg - 1]
        bottom_right = filter_matrix[-num_neg + 1:, -(num_pos + num_neg - 1):]

        # Indexing wrong when there are no negative indexed coefficients
        if num_neg == 1:
            bottom_left = np.zeros((0, 0), dtype=np.float32)
            bottom_right = np.zeros((0, 0), dtype=np.float32)

        return top_left, top_right, bottom_left, bottom_right

class Wavelet:
    """
    Class representing a wavelet.

    Attributes:
        decomp_lp (Filter):    Filter coefficients for decomposition low pass filter
        decomp_hp (Filter):    Filter coefficients for decomposition high pass filter
        recon_lp (Filter):     Filter coefficients for reconstruction low pass filter
        recon_hp (Filter):     Filter coefficients for reconstruction high pass filter
    """
    def __init__(self, decomp_lp, decomp_hp, recon_lp, recon_hp):
        """
        Create a new wavelet based on specified filters

        Args:
            decomp_lp (Filter):    Filter coefficients for decomposition low pass filter
            decomp_hp (Filter):    Filter coefficients for decomposition high pass filter
            recon_lp (Filter):     Filter coefficients for reconstruction low pass filter
            recon_hp (Filter):     Filter coefficients for reconstruction high pass filter
        """
        self.decomp_lp = decomp_lp
        self.decomp_hp = decomp_hp
        self.recon_lp = recon_lp
        self.recon_hp = recon_hp           

def cyclic_conv1d(input_node, filter_):
    """
    Cyclic convolution

    Args:
        input_node:  Input signal (3-tensor [batch, width, in_channels])
        filter_:     Filter

    Returns:
        Tensor with the result of a periodic convolution
    """
    # Create shorthands for TF nodes
    kernel_node = filter_.coeffs
    tl_node, tr_node, bl_node, br_node = filter_.edge_matrices

    # Do inner convolution
    rows,columns,nb_channels = input_node.shape
    input_node_resh = tf.transpose(input_node, perm=[2,0,1])
    input_node_resh = tf.reshape(input_node_resh,[rows*nb_channels,columns,1])
    inner_resh = tf.nn.conv1d(input_node_resh, kernel_node[::-1], stride=1, padding='VALID')
    inner_resh = tf.reshape(inner_resh,[nb_channels,rows,inner_resh.shape[1]])
    inner = tf.transpose(inner_resh, perm=[1,2,0])

    # Create shorthands for shapes
    input_shape = tf.shape(input_node)
    tl_shape = tf.shape(tl_node)
    tr_shape = tf.shape(tr_node)
    bl_shape = tf.shape(bl_node)
    br_shape = tf.shape(br_node)

    # Slices of the input signal corresponding to the corners
    tl_slice = tf.slice(input_node,
                        [0, 0, 0],
                        [-1, tl_shape[2], -1])
    tr_slice = tf.slice(input_node,
                        [0, input_shape[1] - tr_shape[2], 0],
                        [-1, tr_shape[2], -1])
    bl_slice = tf.slice(input_node,
                        [0, 0, 0],
                        [-1, bl_shape[2], -1])
    br_slice = tf.slice(input_node,
                        [0, input_shape[1] - br_shape[2], 0],
                        [-1, br_shape[2], -1])

    # TODO: It just werks (It's the magic of the algorithm). i.e. Why do we have to transpose?
    tl_node = tf.tile(tl_node, [nb_channels,1,1])
    tr_node = tf.tile(tr_node, [nb_channels,1,1])
    bl_node = tf.tile(bl_node, [nb_channels,1,1])
    br_node = tf.tile(br_node, [nb_channels,1,1])
    tl = tl_node @ tf.transpose(tl_slice, perm=[2, 1, 0])
    tr = tr_node @ tf.transpose(tr_slice, perm=[2, 1, 0])
    bl = bl_node @ tf.transpose(bl_slice, perm=[2, 1, 0])
    br = br_node @ tf.transpose(br_slice, perm=[2, 1, 0])

    head = tf.transpose(tl + tr, perm=[2, 1, 0])
    tail = tf.transpose(bl + br, perm=[2, 1, 0])

    return tf.concat((head, inner, tail), axis=1)

def upsample(input_node, odd=False):
    """Upsamples. Doubles the length of the input, filling with zeros

    Args:
        input_node: 3-tensor [batch, spatial dim, channels] to be upsampled
        odd:        Bool, optional. If True, content of input_node will be
                    placed on the odd indices of the output. Otherwise, the
                    content will be placed on the even indeces. This is the
                    default behaviour.

    Returns:
        The upsampled output Tensor.
    """

    columns = []
    for col in tf.unstack(input_node, axis=1):
        columns.extend([tf.expand_dims(col,1), tf.expand_dims(tf.zeros_like(col),1)])
        
    if odd:
        # https://stackoverflow.com/questions/30097512/how-to-perform-a-pairwise-swap-of-a-list
        l = len(columns) & -2
        columns[1:l:2], columns[:l:2] = columns[:l:2], columns[1:l:2]

    return tf.concat(columns, 1)

def dwt1d(input_node, wavelet, levels=1):
    """
    Constructs a TF computational graph computing the 1D DWT of an input signal.

    Args:
        input_node:     A 3D tensor containing the signal. The dimensions should be
                        [batch, signal, channels].
        wavelet:        Wavelet object
        levels:         Number of levels.

    Returns:
        The output node of the DWT graph.
    """
    coeffs = [None] * (levels + 1)

    last_level = input_node

    for level in range(levels):
        lp_res = cyclic_conv1d(last_level, wavelet.decomp_lp)[:, ::2, :]
        hp_res = cyclic_conv1d(last_level, wavelet.decomp_hp)[:, 1::2, :]

        last_level = lp_res
        coeffs[levels - level] = hp_res

    coeffs[0] = last_level
    return tf.concat(coeffs, axis=1)

def dwt2d(input_node, wavelet, levels=1):
    """
    Constructs a TF computational graph computing the 2D DWT of an input signal.

    Args:
        input_node:     A 3D tensor containing the signal. The dimensions should be
                        [rows, cols, channels].
        wavelet:        Wavelet object.
        levels:         Number of levels.

    Returns:
        The output node of the DWT graph.
    """
    # TODO: Check that level is a reasonable number
    # TODO: Check types

    coeffs = [None] * levels
    
    last_level = input_node
    m, n = int(input_node.shape[0]), int(input_node.shape[1])

    for level in range(levels):
        local_m, local_n = m // (2 ** level), n // (2 ** level)

        first_pass = dwt1d(last_level, wavelet, 1)
        second_pass = tf.transpose(
            dwt1d(
                tf.transpose(first_pass, perm=[1, 0, 2]),
                wavelet,
                1
            ),
            perm=[1, 0, 2]
        )

        last_level = tf.slice(second_pass, [0, 0, 0], [local_m // 2, local_n // 2, -1])
        coeffs[level] = [
            tf.slice(second_pass, [local_m // 2, 0, 0], [local_m // 2, local_n // 2, -1]),
            tf.slice(second_pass, [0, local_n // 2, 0], [local_m // 2, local_n // 2, -1]),
            tf.slice(second_pass, [local_m // 2, local_n // 2, 0],
                     [local_m // 2, local_n // 2, -1])
        ]

    for level in range(levels - 1, -1, -1):
        upper_half = tf.concat([last_level, coeffs[level][0]], 0)
        lower_half = tf.concat([coeffs[level][1], coeffs[level][2]], 0)

        last_level = tf.concat([upper_half, lower_half], 1)

    return last_level

def idwt1d(input_node, wavelet, levels=1):
    """
    Constructs a TF graph that computes the 1D inverse DWT for a given wavelet.

    Args:
        input_node (tf.placeholder):             Input signal. A 3D tensor with dimensions
                                                 as [batch, signal, channels]
        wavelet (tfwavelets.dwtcoeffs.Wavelet):  Wavelet object.
        levels (int):                            Number of levels.

    Returns:
        Output node of IDWT graph.
    """
    m, n = int(input_node.shape[0]), int(input_node.shape[1])

    first_n = n // (2 ** levels)
    last_level = tf.slice(input_node, [0, 0, 0], [m, first_n, -1])

    for level in range(levels - 1, -1 , -1):
        local_n = n // (2 ** level)

        detail = tf.slice(input_node, [0, local_n//2, 0], [m, local_n//2, -1])

        lowres_padded = upsample(last_level, odd=False)
        detail_padded = upsample(detail, odd=True)

        lowres_filtered = cyclic_conv1d(lowres_padded, wavelet.recon_lp)
        detail_filtered = cyclic_conv1d(detail_padded, wavelet.recon_hp)

        last_level = lowres_filtered + detail_filtered

    return last_level

def idwt2d(input_node, wavelet, levels=1):
    """
    Constructs a TF graph that computes the 2D inverse DWT for a given wavelet.

    Args:
        input_node (tf.placeholder):             Input signal. A 3D tensor with dimensions
                                                 as [rows, cols, channels]
        wavelet (tfwavelets.dwtcoeffs.Wavelet):  Wavelet object.
        levels (int):                            Number of levels.

    Returns:
        Output node of IDWT graph.
    """
    m, n = int(input_node.shape[0]), int(input_node.shape[1])
    first_m, first_n = m // (2 ** levels), n // (2 ** levels)

    last_level = tf.slice(input_node, [0, 0, 0], [first_m, first_n, -1])

    for level in range(levels - 1, -1, -1):
        local_m, local_n = m // (2 ** level), n // (2 ** level)

        # Extract detail spaces
        detail_tr = tf.slice(input_node, [local_m // 2, 0, 0],
                             [local_n // 2, local_m // 2, -1])
        detail_bl = tf.slice(input_node, [0, local_n // 2, 0],
                             [local_n // 2, local_m // 2, -1])
        detail_br = tf.slice(input_node, [local_n // 2, local_m // 2, 0],
                             [local_n // 2, local_m // 2, -1])

        # Construct image of this DWT level
        upper_half = tf.concat([last_level, detail_tr], 0)
        lower_half = tf.concat([detail_bl, detail_br], 0)

        this_level = tf.concat([upper_half, lower_half], 1)

        # First pass, corresponding to second pass in dwt2d
        first_pass = tf.transpose(
            idwt1d(
                tf.transpose(this_level, perm=[1, 0, 2]),
                wavelet,
                1
            ),
            perm=[1, 0, 2]
        )
        # Second pass, corresponding to first pass in dwt2d
        second_pass = idwt1d(first_pass, wavelet, 1)

        last_level = second_pass

    return last_level
    
def create_wavelet_transform_filters(wavelet_type, tf_prec,np_prec):
    """    
    Arguments:
    wavelet -- 'haar', 'db2'               
    
    Returns:
    filter - filter useful for the wavelet transform
    """
    if wavelet_type=='haar':
        # Haar wavelet
        c0 = np.sqrt(2)/2
        return Wavelet(
            Filter(np.array([c0, c0]), 1, tf_prec,np_prec),
            Filter(np.array([-c0, c0]), 0, tf_prec,np_prec),
            Filter(np.array([c0, c0]), 0, tf_prec,np_prec),
            Filter(np.array([c0, -c0]), 1, tf_prec,np_prec),
         )
    elif wavelet_type=='db2':
        # Daubechies wavelets    
        c1 = (1+np.sqrt(3))/(4*np.sqrt(2))
        c2 = (3+np.sqrt(3))/(4*np.sqrt(2))
        c3 = (3-np.sqrt(3))/(4*np.sqrt(2))
        c4 = (1-np.sqrt(3))/(4*np.sqrt(2))
        
        return Wavelet(
            Filter(np.array([c4, c3, c2, c1]), 3, tf_prec,np_prec),
            Filter(np.array([-c1, c2, -c3, c4]), 0, tf_prec,np_prec),
            Filter(np.array([c1, c2, c3, c4]), 0, tf_prec,np_prec),
            Filter(np.array([c4, -c3, c2, -c1]), 3, tf_prec,np_prec)
        )
