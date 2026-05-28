import numpy as np
import random
import paddle
import math
from PIL import Image
from PIL import ImageEnhance

from clip import tokenize, load_model
from paddle.vision import transforms

_, clip_transforms = load_model('ViT_B_32', pretrained=True)

try:
    import accimage
except ImportError:
    accimage = None


class SpatialTransform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        pass

    def randomize_parameters(self):
        pass


class CenterCornerCrop(SpatialTransform):
    """
    Crops the given PIL.Image at the center or the corner.
    """

    def __init__(self, size, crop_position):
        """
        :param size: int
        Desired output size of the crop. Only square crop is supported.
        :param crop_position: str
        Must be one of ['c', 'tl', 'tr', 'bl', 'br']
        """
        super(CenterCornerCrop, self).__init__()
        self.size = size
        self.crop_position = crop_position

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]
        x1 = y1 = x2 = y2 = 0
        if self.crop_position == 'c':
            center_x = round(image_width / 2.)
            center_y = round(image_height / 2.)
            box_half = round(self.size / 2.)
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height
        return img.crop((x1, y1, x2, y2))


class RandomHorizontalFlip(SpatialTransform):
    def __init__(self, prob=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob
        self.randomized_parameters()

    def __call__(self, img):
        """
        :param img: PIL.Image
        Image to be flipped
        :return: PIL.Image
        Randomly flipped image
        """
        if self.p < self.prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img

    def randomized_parameters(self):
        self.p = random.random()


class RandomCenterCornerCrop(SpatialTransform):
    def __init__(self,
        size,
        scale=(0.64, 1.0),
        ratio=(3.0 / 4, 4.0 / 3),
        interpolation=Image.BICUBIC
    ):
        super(RandomCenterCornerCrop, self).__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        # self.randomize_parameters()

    def randomize_parameters(self, img):
        width, height = img.size[0], img.size[1]
        area = height * width

        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = tuple(math.log(x) for x in self.ratio)
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                self.i = random.randint(0, height - h)
                self.j = random.randint(0, width - w)
                self.h = h
                self.w = w
                return 

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            # return whole image
            w = width
            h = height
        self.i = (height - h) // 2
        self.j = (width - w) // 2
        self.h = h
        self.w = w

    def __call__(self, img):
        img = img.crop((self.j, self.i, self.j + self.w, self.i + self.h))
        return img.resize((self.size, self.size), self.interpolation)

class Compose(SpatialTransform):
    """
    Composed several transforms together.
    :param list
    List of transforms to Compose
    """

    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(SpatialTransform):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255]
    to a paddle.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    """

    def __init__(self, norm_value=255):
        super(ToTensor, self).__init__()
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        :param pic: [PIL.Image or numpy.ndarray]. Image to be converted to tensor.
        :return: [Tensor]. Converted image.
        """
        if isinstance(pic, np.ndarray):
            img = paddle.to_tensor(pic.transpose((2, 0, 1))).astype(np.float32)
            return img.div_(self.norm_value)
        if accimage is not None and isinstance(pic, accimage.Image):
            assert True, "ToTensor fails: accimage"
        # handle PIL Image
        if pic.mode != "RGB":
            assert True, "ToTensor fails: PIL Image is not RGB"
        else:
            return paddle.vision.transforms.to_tensor(pic)


class Scale(SpatialTransform):
    """
    Rescale the input PIL.Image to the given size.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        :param size: sequence or int
        Desired output size. If size is a sequence like (w, h), output size will be matched to this. If size is an
        int, smaller edge of the image will be matched to this number, i.e. if height > width, then image will be
        rescaled to (size * height / width, size)
        :param interpolation: optional
        Desired interpolation. Default is ``PIL.Image.BILINEAR``
        """
        super(Scale, self).__init__()
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        :param img: PIL.Image
        Image to be scaled
        :return: PIL.Image
        Rescaled Image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if w <= h and w == self.size or h <= w and h == self.size:
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class HorizontalFlip(SpatialTransform):
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def __call__(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)


class RandomApply(SpatialTransform):
    def __init__(self, transform, prob=0.5):
        super(RandomApply, self).__init__()
        self.transform = transform
        self.prob = prob
        self.randomize_parameters()

    def __call__(self, img):
        if self.p < self.prob:
            return self.transform(img)
        else:
            return img

    def randomize_parameters(self, recursive=True):
        self.p = random.random()
        if recursive and hasattr(self.transform, 'randomize_parameters'):
            self.transform.randomize_parameters()


class RandomChoice(SpatialTransform):
    def __init__(self, transforms):
        super(RandomChoice, self).__init__()
        self.transforms = transforms
        assert len(transforms) > 0
        self.randomize_parameters()

    def __call__(self, img):
        return self.transfrom_to_apply(img)

    def randomize_parameters(self, recursive=True):
        self.transfrom_to_apply = self.transforms[random.randint(0, len(self.transforms) - 1)]
        if recursive and hasattr(self.transfrom_to_apply, 'randomize_parameters'):
            self.transfrom_to_apply.randomize_parameters()


class BrightnessJitter(SpatialTransform):
    def __init__(self, brightness=0.5):
        super(BrightnessJitter, self).__init__()
        self.brightness = brightness
        self.randomize_parameters()

    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.factor)

    def randomize_parameters(self):
        self.factor = random.uniform(self.brightness, 1.0)


class RandomRotation(SpatialTransform):
    def __init__(self, degrees=20, interpolation=Image.BILINEAR):
        super(RandomRotation, self).__init__()
        self.degrees = degrees
        self.interpolation = interpolation
        self.randomize_parameters()

    def __call__(self, img: Image.Image):
        return img.rotate(self.angle, self.interpolation)

    def randomize_parameters(self):
        self.angle = random.uniform(-self.degrees, self.degrees)

class RandomGrayscale(SpatialTransform):
    def __init__(self, p=0.2):
        super(RandomGrayscale, self).__init__()
        self.p = p
        self.grayscale = transforms.Grayscale(num_output_channels=3)    
    def __call__(self, img):
        r = random.random()
        if r < self.p:
            img = self.grayscale(img)
        return img

class Preprocessing(SpatialTransform):
    def __init__(self, size, degrees=20, brightness=0.5, is_aug=True, center=False):
        super(Preprocessing, self).__init__()
        self.is_aug = is_aug
        self.center = center
        self.f1_1 = RandomCenterCornerCrop(size)
        # self.f1_3 = RandomCenterCornerCrop(size)
        self.f1_2 = CenterCornerCrop(size, 'c')
        self.f2 = RandomApply(
            RandomChoice([
                HorizontalFlip(),
                RandomRotation(degrees),
                BrightnessJitter(brightness),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ]),
            prob=0.3
        )
        self.f3 = RandomGrayscale(0.2)
        self.f4 = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __call__(self, img):
        if not self.center:
            img = self.f1_1(img)
        else:
            img = self.f1_2(img)
        # if self.is_aug:
        #     img = self.f3(self.f2(img))
        img = self.f4(img)
        return img

    def randomize_parameters(self, img):
        self.f1_1.randomize_parameters(img)
        if self.is_aug:
            self.f2.randomize_parameters()
