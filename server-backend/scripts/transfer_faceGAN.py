
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import time
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import dataloader, dataset
from collections import OrderedDict
from torch.autograd import Variable
from pathlib import Path
from tqdm import tqdm
import cv2

# In[2]:


pix2pixhd_dir = Path('../src/pix2pixHD/')

import sys
sys.path.append(str(pix2pixhd_dir))

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html


# In[4]:


with open('../data/test_opt.pkl', mode='rb') as f:
    opt = pickle.load(f)

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')


# In[5]:


opt.batchSize = 4
opt.loadSize = 480
opt.label_nc = 25
# opt.name = 'yaosyBody'
#opt.lr = 7e-4
opt.no_flip = True

opt.model = 'pix2pixHD_faceGAN'
opt.dataroot='../data/source/stocking_dance/'
opt.name = 'gray_maskgirl'
opt.add_face_disc = True
opt.fineSize = opt.loadSize
opt.num_D = 3
# opt.n_local_enhancers = 0
# opt.n_layers_D = 1
# opt.lr = 5e-5
opt.temp_data = False
#opt.no_ganFeat_loss = True
#opt.niter=10
#opt.niter_decay=10
opt.label_nc = 25
# opt.debug = True
#opt.load_pretrain = '../checkpoints/target/'
print(opt)


# In[6]:


def model_inference(inputs, params):
    """
    inputs: a list contains imgs e.g. [img1, img2, ...]
    params: parameters to change
    """
    num_input = len(inputs)
    data_label = torch.Tensor(inputs)
    data_label = tensor_inputs.unsqueeze(1)
    data_inst = torch.zeros(num_input)
    data = {'label': data_label, 'inst': data_inst}
# In[7]:
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)


for i, data in enumerate(tqdm(dataset)):
    generated = model.inference(data['label'], data['inst'])
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()
