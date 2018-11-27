
# coding: utf-8

# In[1]:


import cv2
from pathlib import Path
import numpy as np
import os


# In[9]:


res = (720, 720)
img_dir = Path('/home/yaosy/Diskb/bdmeet/yaosy_slowMov/yaosy_mov')
img_dir.mkdir(exist_ok=True)
video_path = '/home/yaosy/Diskb/bdmeet/Target 1.mp4'

cap = cv2.VideoCapture(video_path)


# In[ ]:


idx = 0
i = -1
while(cap.isOpened()):
    flag, img = cap.read()
    i += 1
    if flag is False:
        break
        
    if i < 40:
        continue    
    shape_dst = np.min(img.shape[:2])
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2

    img = img[oh:oh+shape_dst, ow:ow+shape_dst]
    # img = cv2.resize(img, res)
    cv2.imwrite(str(img_dir.joinpath(f'img_{idx:06d}.png')), img)
    idx += 1


# In[5]:


flag

