# ============================================================
# THIS FILE CONTAINS THE METHODS FOR TRANSFORMING THE IMAGES.
# Authors: Mark Edward M. Gonzales & Lorene C. Uy
# ============================================================

import random

from PIL import Image

# =====================================================
# Class for composing several transformations together
# =====================================================
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, edge, mask):
        assert img.size == mask.size and edge.size == mask.size
        for t in self.transforms:
            img, edge, mask = t(img, edge, mask)
        return img, edge, mask

# ==================================
# Apply random horizontal flipping.
# ==================================
class RandomHorizontallyFlip(object):
    def __call__(self, img, edge, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, edge, mask

# ================
# Apply resizing.
# ================
class Resize(object):
    def __init__(self, size):
        # Reverse since size follows (height, width) while PIL requires (width, height)
        self.size = tuple(reversed(size))

    def __call__(self, img, edge, mask):
        assert img.size == mask.size and edge.size == mask.size
        return img.resize(self.size, Image.BILINEAR), edge.resize(self.size, Image.NEAREST), mask.resize(self.size, Image.NEAREST)