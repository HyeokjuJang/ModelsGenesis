import numpy as np
from PIL import Image
import os

save_path = 'data'
data = np.load("generated_cubes/bat_32_64x64x64_0.npy")

crop_data = data[0,:,:,:] * 255
for i in range(crop_data.shape[1]):
    a = (np.rot90(crop_data[:, i, :], 1)).astype(np.uint8)
    im = Image.fromarray(a)
    im.save(os.path.join(save_path, '{}.png'.format(str(i))))