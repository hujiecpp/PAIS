import copy
import cv2
import mmcv
import numpy as np


def get_erase_cycle(n_iterations):
    if isinstance(n_iterations, int):
        n_iterations = n_iterations
    else:
        assert (
            isinstance(n_iterations, (tuple, list))
            and len(n_iterations) == 2
        )
        n_iterations = np.random.randint(*n_iterations)
    return n_iterations

def get_patch_size(size, h, w, squared):
    if isinstance(size, float):
        assert 0 < size < 1
        return int(size * h), int(size * w)
    else:
        assert isinstance(size, (tuple, list))
        assert len(size) == 2
        assert 0 <= size[0] < 1 and 0 <= size[1] < 1
        w_ratio = np.random.random() * (size[1] - size[0]) + size[0]
        h_ratio = w_ratio

        if not squared:
            h_ratio = (
                np.random.random() * (size[1] - size[0]) + size[0]
            )
        return int(h_ratio * h), int(w_ratio * w)

def erase_pseudo_mask(masks, patch, fill_val=0):
    assert isinstance(masks, BitmapMasks)==1  
    x1, y1, x2, y2 = patch
    tmp = masks.masks.copy()
    tmp[:, y1:y2, x1:x2] = fill_val
    masks = BitmapMasks(tmp, masks.height, masks.width)
    return masks

def erase_pre_mask(masks, patch, fill_val=0):
    x1, y1, x2, y2 = patch
    masks[:, y1:y2, x1:x2] = fill_val
    return masks


def cut_patch(h, 
              w, 
              n_iterations=None, 
              size=None, 
              squared: bool = True):
    n_iterations = get_erase_cycle(n_iterations)
    patches = []
    for i in range(n_iterations):
        # random sample patch size in the image
        ph, pw = get_patch_size(size, h, w, squared)
        # random sample patch left top in the image
        px, py = np.random.randint(0, w - pw), np.random.randint(0, h - ph)
        patches.append([px, py, px + pw, py + ph])
    return patches

def cutout(pre_mask, 
           pseudo_mask, 
           patches):
    for patch in patches:
        x1, y1, x2, y2 = patch
        pre_mask[:, y1:y2, x1:x2] = 0
        pseudo_mask[:, y1:y2, x1:x2] = 0
    return pre_mask, pseudo_mask


"""
def cutout(pre_mask, 
           pseudo_mask, 
           n_iterations=None, 
           size=None, 
           squared: bool = True):
    n_iterations = get_erase_cycle(n_iterations)
    patches = []
    
    b, h, w = pre_mask.shape
    
#---------------------------------#
    print("cutout"+"="*50)
    print(b)
    
    for i in range(n_iterations):
        # random sample patch size in the image
        ph, pw = get_patch_size(size, h, w, squared)
        # random sample patch left top in the image
        px, py = np.random.randint(0, w - pw), np.random.randint(0, h - ph)
        patches.append([px, py, px + pw, py + ph])
    for patch in patches:
        pseudo_mask_cut = erase_pseudo_mask(pseudo_mask, patch, fill_val=0)
        pre_mask_cut = erase_pre_mask(pre_mask, patch, fill_val=0)
    
    print("type(pre_mask_cut)", type(pre_mask_cut))
    print("type(pseudo_mask_cut)", type(pseudo_mask_cut))
    
    return pre_mask_cut, pseudo_mask_cut
"""

