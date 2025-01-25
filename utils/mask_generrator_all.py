
import numpy as np
from PIL import Image, ImageDraw
import math
import random
import torch
from abc import ABC, abstractmethod
#import tensorflow as tf
from torch.nn import functional as F
from torch import nn
np.random.seed(10)
from torch.nn import functional as F
#from util.img_utils import Blurkernel, fft2_m

'''
def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.
    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.
    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data




def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


'''
def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, H, W, C = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t =  np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0
    mask = 1 - mask
    #Fixed mid box
    #mask[..., t:t+h, l:l+w] = 0
    return mask, t, t+h, l, l+w

def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        ##comment the following line for lower masking ratios
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def BatchRandomMask(batch_size, s, hole_range=[0, 1]):
    return np.stack([RandomMask(s, hole_range=hole_range) for _ in range(batch_size)], axis = 0)

def random_mask_92( n=100):
    masks = torch.zeros(100,256,256)
    for i in range(n):
        img = torch.zeros(1,256,256)
        mask =  img.flatten()
        drop_percentage = 70
        num_elements_to_drop = int(len(mask) * drop_percentage / 100)

        # Generate random indices to drop
        indices_to_drop = np.random.choice(len(mask), num_elements_to_drop, replace=False)

        # Set the selected elements to NaN
        mask[indices_to_drop] = 1 #np.nan
        mask = 1 - mask
        masks[i] = mask.reshape(img.shape)
        # Reshape the array back to its original shape
    return masks


def random_rotation(shape):
    cutoff = 100 #was 30
    (n , channels, p, q) = shape
    mask = np.zeros((n,p,q))

    for i in range(n):
        angle = np.random.choice(360, 1)
        mask_one = np.ones((p+cutoff,q+cutoff))
        mask_one[int((p+cutoff)/2):,:] = 0

        im = Image.fromarray(mask_one)
        im = im.rotate(angle)

        left = (p+cutoff - p)/2
        top = (q+cutoff - q)/2
        right = (p+cutoff + p)/2
        bottom = (q+cutoff + q)/2

        # Crop the center of the image
        im = im.crop((left, top, right, bottom))

        mask[i] = np.array(im)

    #mask = np.repeat(mask.reshape([n,1,p,q]), channels, axis=1)
    mask = mask.reshape([n,1,p,q])
    return mask

class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'half', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh
    
    def generate_center_mask(self, shape):
        assert len(shape) == 2
        assert shape[1] % 2 == 0
        center = shape[0] // 2
        center_size = shape[0] // 4
        half_resol = center_size // 2  # for now
        ret = torch.zeros(shape, dtype=torch.float32)
        ret[
            center - half_resol: center + half_resol,
            center - half_resol: center + half_resol,
        ] = 1
        ret = ret.unsqueeze(0).unsqueeze(0)
        return ret

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = BatchRandomMask(1, self.image_size, hole_range=self.mask_prob_range) #self._retrieve_random(img)
            return mask
        elif self.mask_type == "half":
            mask = random_rotation((1, 3, self.image_size, self.image_size))
        elif self.mask_type == 'box':
            #mask, t, th, w, wl = self._retrieve_box(img)
            mask = self.generate_center_mask((self.image_size,self.image_size)) # self._retrieve_box(img)
            return mask #.permute(0,3,1,2)
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask
        elif self.mask_type == 'small':
            img = torch.zeros(1,256,256)
            mask =  img.flatten()
            drop_percentage = 92
            num_elements_to_drop = int(len(mask) * drop_percentage / 100)

            # Generate random indices to drop
            indices_to_drop = np.random.choice(len(mask), num_elements_to_drop, replace=False)

            # Set the selected elements to NaN
            mask[indices_to_drop] = 1 #np.nan
            mask = 1 - mask
            # Reshape the array back to its original shape
            mask = random_mask_92()
            return mask


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)



import scipy
#from motionblur.motionblur import Kernel

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k

@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        
        self.conv.eval()

    def forward(self, data, **kwargs):
        return self.conv(data.float())

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


from functools import partial
from torch import nn
#from util.resizer import Resizer

__NOISE__ = {}

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper
class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

# These next functions are all interpolation methods. x is the distance from the left pixel center

def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))
class Resizer(nn.Module):
    def __init__(self, in_shape, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
        super(Resizer, self).__init__()

        # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
        scale_factor, output_shape = self.fix_scale_and_size(in_shape, output_shape, scale_factor)

        # Choose interpolation method, each method has the matching kernel size
        method, kernel_width = {
            "cubic": (cubic, 4.0),
            "lanczos2": (lanczos2, 4.0),
            "lanczos3": (lanczos3, 6.0),
            "box": (box, 1.0),
            "linear": (linear, 2.0),
            None: (cubic, 4.0)  # set default interpolation method as cubic
        }.get(kernel)

        # Antialiasing is only used when downscaling
        antialiasing *= (np.any(np.array(scale_factor) < 1))

        # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
        sorted_dims = np.argsort(np.array(scale_factor))
        self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

        # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
        field_of_view_list = []
        weights_list = []
        for dim in self.sorted_dims:
            # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
            # weights that multiply the values there to get its result.
            weights, field_of_view = self.contributions(in_shape[dim], output_shape[dim], scale_factor[dim], method,
                                                        kernel_width, antialiasing)

            # convert to torch tensor
            weights = torch.tensor(weights.T, dtype=torch.float32)

            # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
            # tmp_im[field_of_view.T], (bsxfun style)
            weights_list.append(
                nn.Parameter(torch.reshape(weights, list(weights.shape) + (len(scale_factor) - 1) * [1]),
                             requires_grad=False))
            field_of_view_list.append(
                nn.Parameter(torch.tensor(field_of_view.T.astype(np.int32), dtype=torch.long), requires_grad=False))

        self.field_of_view = nn.ParameterList(field_of_view_list)
        self.weights = nn.ParameterList(weights_list)

    def forward(self, in_tensor):
        x = in_tensor

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):
            # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
            x = torch.transpose(x, dim, 0)

            # This is a bit of a complicated multiplication: x[field_of_view.T] is a tensor of order image_dims+1.
            # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
            # only, this is why it only adds 1 dim to 5the shape). We then multiply, for each pixel, its set of positions with
            # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
            # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
            # same number
            x = torch.sum(x[fov] * w, dim=0)

            # Finally we swap back the axes to the original order
            x = torch.transpose(x, dim, 0)

        return x

    def fix_scale_and_size(self, input_shape, output_shape, scale_factor):
        # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
        # same size as the number of input dimensions)
        if scale_factor is not None:
            # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
            if np.isscalar(scale_factor) and len(input_shape) > 1:
                scale_factor = [scale_factor, scale_factor]

            # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
            scale_factor = list(scale_factor)
            scale_factor = [1] * (len(input_shape) - len(scale_factor)) + scale_factor

        # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
        # to all the unspecified dimensions
        if output_shape is not None:
            output_shape = list(input_shape[len(output_shape):]) + list(np.uint(np.array(output_shape)))

        # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
        # sub-optimal, because there can be different scales to the same output-shape.
        if scale_factor is None:
            scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

        # Dealing with missing output-shape. calculating according to scale-factor
        if output_shape is None:
            output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

        return scale_factor, output_shape

    def contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
        # such that each position from the field_of_view will be multiplied with a matching filter from the
        # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
        # around it. This is only done for one dimension of the image.

        # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
        # 1/sf. this means filtering is more 'low-pass filter'.
        fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
        kernel_width *= 1.0 / scale if antialiasing else 1.0

        # These are the coordinates of the output image
        out_coordinates = np.arange(1, out_length + 1)

        # since both scale-factor and output size can be provided simulatneously, perserving the center of the image requires shifting
        # the output coordinates. the deviation is because out_length doesn't necesary equal in_length*scale.
        # to keep the center we need to subtract half of this deivation so that we get equal margins for boths sides and center is preserved.
        shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2

        # These are the matching positions of the output-coordinates on the input image coordinates.
        # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
        # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
        # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
        # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
        # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
        # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
        # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
        # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
        match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)

        # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
        left_boundary = np.floor(match_coordinates - kernel_width / 2)

        # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
        # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
        expanded_kernel_width = np.ceil(kernel_width) + 2

        # Determine a set of field_of_view for each each output position, these are the pixels in the input image
        # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
        # vertical dim is the pixels it 'sees' (kernel_size + 2)
        field_of_view = np.squeeze(
            np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

        # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
        # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
        # 'field_of_view')
        weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

        # Normalize weights to sum up to 1. be careful from dividing by 0
        sum_weights = np.sum(weights, axis=1)
        sum_weights[sum_weights == 0] = 1.0
        weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

        # We use this mirror structure as a trick for reflection padding at the boundaries
        mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

        # Get rid of  weights and pixel positions that are of zero weight
        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        weights = np.squeeze(weights[:, non_zero_out_pixels])
        field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

        # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
        return weights, field_of_view


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

'''
def tf_mask_generator(s, tf_hole_range):
    def random_mask_generator(hole_range):
        while True:
            yield RandomMask(s, hole_range=hole_range)
    return tf.data.Dataset.from_generator(random_mask_generator, tf.float32, tf.TensorShape([1, s, s]), (tf_hole_range,))
'''
import yaml


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(data.shape[0], 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred



@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude
