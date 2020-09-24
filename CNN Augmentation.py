#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:37:29 2020

@author: siddharthsmac
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


datagen = ImageDataGenerator(
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = 'nearest')


img = load_img('/users/siddharthsmac/desktop/IMG_0849.jpeg')

x = img_to_array(img)
x = x.reshape((1,)+x.shape)

i = 0

for batch in datagen.flow(x, batch_size = 1, save_to_dir = '/users/siddharthsmac/desktop/practice_preview', save_prefix = 'Cat', save_format = 'jpeg'):
    i+=1
    if i > 50:
        break