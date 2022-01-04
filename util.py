import numpy as np
import cv2 
#from scipy.misc import imread, imresize, imsave
import torch


def load_image(filepath):
    image = cv2.imread(filepath)
    #cv2.imshow("1",image)
    #cv2.waitKey(0)
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    #min = image.min()
    #max = image.max()
    image = torch.FloatTensor(image.size()).copy_(image)
    #image.add_(-min).mul_(1.0 / (max - min))
    image.mul_(1.0/255.0)
    image = image.mul_(2).add_(-1)
    return image


def save_image(image, filename):
    #print(image)
    image_out = image.add(1).div(2)
    image_out = image_out.numpy()
    image_out *= 255.0
    image_out = image_out.clip(0, 255)
    image_out = np.transpose(image_out, (1, 2, 0))
    image_out = image_out.astype(np.uint8)
    r,g,b = cv2.split(image_out)
    image_out = cv2.merge([b,g,r])
    #cv2.imshow("1",image_out)
    #cv2.waitKey(0)
    cv2.imwrite(filename, image_out)
    print ("Image saved as {}".format(filename))

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg"])
