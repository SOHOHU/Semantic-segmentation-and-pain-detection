 # -*- coding: utf-8 -*-
import random
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as tf
import shutil
from labelme import utils
import cv2

 
class Augmentation:
    def __init__(self):
        pass
    def rotate(self,image,mask,angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-45, 45]) # -180~180随机选一个角度旋转
        if isinstance(angle,list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def flip(self,image,mask): #水平翻转和垂直翻转
        if random.random()>0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        if random.random()<0.5:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def randomResizeCrop(self,image,mask,scale=(0.3,1.0),ratio=(1,1)):#scale表示随机crop出来的图片会在的0.3倍至1倍之间，ratio表示长宽比
        img = np.array(image)
        h_image, w_image, c = img.shape
        resize_size = h_image
        # image = np.float(image)
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def adjustContrast(self,image,mask):
        factor = transforms.RandomRotation.get_params([0,10])   #这里调增广后的数据的对比度
        image = tf.adjust_contrast(image,factor)
        #mask = tf.adjust_contrast(mask,factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image,mask
    def adjustBrightness(self,image,mask):
        factor = transforms.RandomRotation.get_params([1, 2])  #这里调增广后的数据亮度
        image = tf.adjust_brightness(image, factor)
        #mask = tf.adjust_contrast(mask, factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
    def centerCrop(self,image,mask,size=None): #中心裁剪
        if size == None:size = image.size   #若不设定size，则是原图。
        image = tf.center_crop(image,size)
        mask = tf.center_crop(mask,size)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image,mask
    def adjustSaturation(self,image,mask): #调整饱和度
        factor = transforms.RandomRotation.get_params([1, 2])  # 这里调增广后的数据亮度
        image = tf.adjust_saturation(image, factor)
        #mask = tf.adjust_saturation(mask, factor)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

 
def augmentationData(image_path, mask_path, img_type, mask_type, option=[1,2,3,4,5,6,7], save_dir=None):
    aug_image_savedDir = os.path.join(save_dir, 'JPEGImages')
    aug_mask_savedDir = os.path.join(save_dir, 'SegmentationClass')
    if not os.path.exists(aug_image_savedDir):
        os.makedirs(aug_image_savedDir)
        print('create aug image dir.....')
    if not os.path.exists(aug_mask_savedDir):
        os.makedirs(aug_mask_savedDir)
        print('create aug mask dir.....')
    aug = Augmentation()
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_path) for f in filenames if os.path.splitext(f)[1].lower() == '.' + img_type]
    masks = [os.path.join(dp, f) for dp, dn, filenames in os.walk(mask_path) for f in filenames if os.path.splitext(f)[1].lower() == '.' + mask_type]
    datas = list(zip(images, masks))

    for image_path, mask_path in datas:
        base_image_name = os.path.splitext(os.path.basename(image_path))[0]
        base_mask_name = os.path.splitext(os.path.basename(mask_path))[0]

        # 复制原始图像和掩码
        shutil.copy(image_path, os.path.join(aug_image_savedDir, base_image_name + '.' + img_type))
        shutil.copy(mask_path, os.path.join(aug_mask_savedDir, base_mask_name + '.' + mask_type))

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        # 对每种选项执行增广操作...
        if 1 in option:
            image_tensor, mask_tensor = aug.rotate(image, mask)
            image_rotate_path = os.path.join(aug_image_savedDir, f'{base_image_name}_rotate.{img_type}')
            mask_rotate_path = os.path.join(aug_mask_savedDir, f'{base_mask_name}_rotate.{mask_type}')
            transforms.ToPILImage()(image_tensor).save(image_rotate_path)
            transforms.ToPILImage()(mask_tensor).save(mask_rotate_path)
            # mask_tensor = mask_tensor.squeeze(0).numpy().astype(np.uint8)
            # print(mask_tensor.dtype, type(mask_tensor))
            
            # utils.lblsave(mask_rotate_path, mask_tensor)
            mask_tensor = cv2.imread(mask_rotate_path, cv2.IMREAD_GRAYSCALE)
            utils.lblsave(mask_rotate_path, mask_tensor)
            
        if 2 in option:
            image_tensor, mask_tensor = aug.flip(image, mask)
            image_rotate_path = os.path.join(aug_image_savedDir, f'{base_image_name}_flip.{img_type}')
            mask_rotate_path = os.path.join(aug_mask_savedDir, f'{base_mask_name}_flip.{mask_type}')
            transforms.ToPILImage()(image_tensor).save(image_rotate_path)
            transforms.ToPILImage()(mask_tensor).save(mask_rotate_path)
            mask_tensor = cv2.imread(mask_rotate_path, cv2.IMREAD_GRAYSCALE)
            utils.lblsave(mask_rotate_path, mask_tensor)
        if 3 in option:
            image_tensor, mask_tensor = aug.randomResizeCrop(image, mask)
            image_rotate_path = os.path.join(aug_image_savedDir, f'{base_image_name}_randomResizeCrop.{img_type}')
            mask_rotate_path = os.path.join(aug_mask_savedDir, f'{base_mask_name}_randomResizeCrop.{mask_type}')
            transforms.ToPILImage()(image_tensor).save(image_rotate_path)
            transforms.ToPILImage()(mask_tensor).save(mask_rotate_path)
            mask_tensor = cv2.imread(mask_rotate_path, cv2.IMREAD_GRAYSCALE)
            utils.lblsave(mask_rotate_path, mask_tensor)
        if 4 in option:
            image_tensor, mask_tensor = aug.adjustContrast(image, mask)
            image_rotate_path = os.path.join(aug_image_savedDir, f'{base_image_name}_adjustContrast.{img_type}')
            mask_rotate_path = os.path.join(aug_mask_savedDir, f'{base_mask_name}_adjustContrast.{mask_type}')
            transforms.ToPILImage()(image_tensor).save(image_rotate_path)
            transforms.ToPILImage()(mask_tensor).save(mask_rotate_path)
            mask_tensor = cv2.imread(mask_rotate_path, cv2.IMREAD_GRAYSCALE)
            utils.lblsave(mask_rotate_path, mask_tensor)
        if 5 in option:
            image_tensor, mask_tensor = aug.centerCrop(image, mask)
            image_rotate_path = os.path.join(aug_image_savedDir, f'{base_image_name}_centerCrop.{img_type}')
            mask_rotate_path = os.path.join(aug_mask_savedDir, f'{base_mask_name}_centerCrop.{mask_type}')
            transforms.ToPILImage()(image_tensor).save(image_rotate_path)
            transforms.ToPILImage()(mask_tensor).save(mask_rotate_path)
            mask_tensor = cv2.imread(mask_rotate_path, cv2.IMREAD_GRAYSCALE)
            utils.lblsave(mask_rotate_path, mask_tensor)
        if 6 in option:
            image_tensor, mask_tensor = aug.adjustBrightness(image, mask)
            image_rotate_path = os.path.join(aug_image_savedDir, f'{base_image_name}_adjustBrightness.{img_type}')
            mask_rotate_path = os.path.join(aug_mask_savedDir, f'{base_mask_name}_adjustBrightness.{mask_type}')
            transforms.ToPILImage()(image_tensor).save(image_rotate_path)
            transforms.ToPILImage()(mask_tensor).save(mask_rotate_path)
            mask_tensor = cv2.imread(mask_rotate_path, cv2.IMREAD_GRAYSCALE)
            utils.lblsave(mask_rotate_path, mask_tensor)
        if 7 in option:
            image_tensor, mask_tensor = aug.adjustSaturation(image, mask)
            image_rotate_path = os.path.join(aug_image_savedDir, f'{base_image_name}_adjustSaturation.{img_type}')
            mask_rotate_path = os.path.join(aug_mask_savedDir, f'{base_mask_name}_adjustSaturation.{mask_type}')
            transforms.ToPILImage()(image_tensor).save(image_rotate_path)
            transforms.ToPILImage()(mask_tensor).save(mask_rotate_path)
            mask_tensor = cv2.imread(mask_rotate_path, cv2.IMREAD_GRAYSCALE)
            utils.lblsave(mask_rotate_path, mask_tensor)
 
 
# if __name__ == "__main__":
train_path = r'datasets/JPEGImages'
groud_truth_path = r'datasets/SegmentationClass'
img_type = 'jpg'
mask_type = 'png'
augmentationData(train_path,groud_truth_path,img_type,mask_type, save_dir='VOCdevkit/VOC2007', option=[1,2])
    
# img_path = 'datasets\JPEGImages'
# label_path = 'datasets\SegmentationClass'
# save_img = 'VOCdevkit\VOC2007\JPEGImages'
# save_label = 'VOCdevkit\VOC2007\SegmentationClass'
