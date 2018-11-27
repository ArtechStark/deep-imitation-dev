import cv2
from pathlib import Path
import numpy as np
import os

targetSize = 480
filepath = Path('./')
savepath = Path('./')
filename_list = os.listdir(str(filepath))
for filename in filename_list:
    img = cv2.imread(str(filepath.joinpath(filename)))
    img = cv2.resize(img, (targetSize, targetSize))
    cv2.imwrite(str(savepath.joinpath(filename)), img)
    print(filename)
