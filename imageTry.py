from PIL import Image
import numpy as np
import random
import sys
import math
np.set_printoptions(threshold=sys.maxsize)


#해당 이미지를 배열로 변환
img = Image.open("/Users/chaewookim/Desktop/ColorClassification/Blue/4.jpg")
pix = np.array(img)

print(img)