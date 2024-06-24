
import numpy as np
from PIL import Image, ImageDraw
import math
import random
import torch
#import tensorflow as tf
np.random.seed(10)
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
        #mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def BatchRandomMask(batch_size, s, hole_range=[0, 1]):
    return np.stack([RandomMask(s, hole_range=hole_range) for _ in range(batch_size)], axis = 0)

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


'''
def tf_mask_generator(s, tf_hole_range):
    def random_mask_generator(hole_range):
        while True:
            yield RandomMask(s, hole_range=hole_range)
    return tf.data.Dataset.from_generator(random_mask_generator, tf.float32, tf.TensorShape([1, s, s]), (tf_hole_range,))
'''