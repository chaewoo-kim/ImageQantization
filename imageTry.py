from PIL import Image
import numpy as np
import random
import sys
import math
import cv2
from rembg import remove
np.set_printoptions(threshold=sys.maxsize)


#해당 이미지를 배열로 변환
img = Image.open("/Users/chaewookim/Desktop/ColorClassification/Black/12.jpg")

out = remove(img)

rgbImage = out.convert('RGB')

pix = np.array(rgbImage)


row,col,floor = pix.shape
twoDim_array = np.column_stack(((np.repeat(np.arange(row), col)), pix.reshape(row*col, -1)))
twoDim_array = np.delete(twoDim_array, 0, axis=1)



file = open("two", 'w+')
toString = ''.join(str(twoDim_array))
file.write(toString)

rgbImage.show()