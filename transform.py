import cv2
import numpy as np
import math

"""
This is a modified copy of:
https://github.com/Hsuxu/Image-preprocessing/blob/master/src/preprocessing.py

This is a image preprocessing library for personal use
Is there any problems please concat:
Hsuxu820@gmail.com
"""


def remove_mean(image):
    """
    remove RGB mean values which from ImageNet
    input
        image:  RGB image np.ndarray 
                type of elements is np.uint8
    return:
        image:  remove RGB mean and scale to [0,1] 
                type of elements is np.float32
    """
    mean = [0.48462227599918,  0.45624044862054, 0.40588363755159]
    image = image.astype(np.float32)
    image = np.subtract(np.divide(image, 255.0), mean)
    return image


def standardize(image, mean=[0.48462227599918,  0.45624044862054, 0.40588363755159], std=[0.22889466674951, 0.22446679341259, 0.22495548344775]):
    """
    standardize RGB mean and std values which from ImageNet
    input:
        image:  RGB image np.ndarray 
                type of elements is np.uint8
    return:
        image:  standarded image
                type of elements is np.float32
    """
    image = image / 255.0
    image = np.divide(np.subtract(image, mean), std, dtype=np.float32)
    return image


def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample image
    Output:
        Normalized sample
    x=1.0*(x-np.min(x))/(np.max(x)-np.min(x))
    """
    data.astype(np.float32)
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))


def contrast_adjust(image, alpha=1.3, beta=20):
    """
    adjust constrast through gamma correction
    newimg = image * alpha + beta
    input:
        image: np.uint8 or np.float32
    output:
        image: np.uint8 or np.float
    """
    newimage = image.astype(np.float32) * alpha + beta

    if type(image[0, 0, 0]) == np.uint8:
        newimage[newimage < 0] = 0
        newimage[newimage > 255] = 255
        return np.uint8(newimage)
    else:
        newimage[newimage < 0] = 0
        newimage[newimage > 1] = 1.
        return newimage


def random_flip(image, lr, ud):
    """
    random flip image 
    """
    if lr:
        if np.random.random() > 0.5:
            image = cv2.flip(image, flipCode=1)
    if ud:
        if np.random.random() > 0.5:
            image = cv2.flip(image, flipCode=0)
    return image


def image_crop(image, crop=None, target_shape=None, random_crop=False):
    """
    if crop is None crop size is generated with a random size range from [0.5*height,height]
    if random_crop == True image croped from a random position
    input:
        image: image np.ndarray [H,W,C]
        crop: [target_height,target_width]
    output:
        croped image with shape[crop[0],crop[1],C]
    """
    hei, wid, _ = image.shape
    if crop is None:
        crop = (np.random.randint(int(hei / 2),  hei),
                np.random.randint(int(wid / 2),  wid))
    th, tw = [int(round(x / 2)) for x in crop]
    if random_crop:
        th, tw = np.random.randint(
            0, hei - crop[0]), np.random.randint(0, wid - crop[1])
    croped_img = image[th:th + crop[0], tw:tw + crop[1]]
    if target_shape:
        t_h, t_w = target_shape
        croped_img = cv2.resize(croped_img, (t_w, t_h), interpolation=cv2.INTER_NEAREST)
    return croped_img



def image_pad(image, pad_width=None, mode='symmetric', keep_shape=False):
    """
    pad an image 
    like np.pad way
    input:
        image: ndarray [rgb]

    """
    hei, wid = image.shape[0], image.shape[1]

    if pad_width is None:
        th = hei//10
        tw = wid//10
        pad_width = ((th, th), (tw, tw), (0, 0))
    if len(image.shape) == 3:
        newimage = np.pad(image, pad_width, mode)
    elif len(image.shape) == 2:
        newimage = np.squeeze(np.pad(image[:, :, np.newaxis], pad_width, mode))

    if keep_shape:
        return cv2.resize(newimage, (wid, hei), interpolation=cv2.INTER_NEAREST)
    return newimage


# end of this copy

def reshape(img, shape):
    # input is (h, w), swap the h and w that suit for PIL
    h, w = shape
    shape = w, h
    img = img.resize(shape)
    return img

def suit_for_min_shape(img, shape):
    min_h, min_w = shape
    img_w, img_h = img.size
    if min_h <= img_h and min_h <= img_h:
        return img
    h_scale = (min_h + 1) / img_h
    w_scale = (min_w + 1) / img_w
    scale = max([1.0, h_scale, w_scale])
    new_h = math.ceil(img_h * scale)
    new_w = math.ceil(img_w * scale)
    return reshape(img, (new_h, new_w))

def random_horizontal_flip(img):
    return random_flip(img, True, None)


if __name__ == "__main__":
    from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
    from data_loader import ImageDataLoader

    def preprocess_image(img):
        img = reshape(img, (256, 128))
        img = img_to_array(img)
        img = image_pad(img, ((10, 10), (10, 10), (0, 0)))
        img = image_crop(img, (256, 128), random_crop=True)
        img = standardize(img)
        return img

    def test_preprocess_image():
        data = ImageDataLoader(
            '../Market/bounding_box_train', transforms=[preprocess_image], batch_size=32)
        data_iter = data.flow()
        batch = next(data_iter)
        img = batch[0][0]
        array_to_img(img).save('test.jpg')
    test_preprocess_image()
