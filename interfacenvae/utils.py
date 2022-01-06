# import and utils for nvae

# this script should be executed before calling the streamlit app
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tensorflow as tf
from skimage import io, transform
from torchvision import datasets, transforms
import os

#check if folder /content/downloaded_imgs exists
tmp_download_folder = "/content/downloaded_imgs"
if not os.path.exists(tmp_download_folder):
  os.mkdir(tmp_download_folder)


# use GPU for computation if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# make sure that files in local directory can be imported
sys.path.insert(0, ".")

# try to import nvae module and download it if it fails
import urllib

# load code for NVAE
try:
    import nvae
except ImportError:
    urllib.request.urlretrieve("https://uu-sml.github.io/course-apml-public/lab/nvae.py", "nvae.py")

# load nvae model celeba_256
[] = nvae.load_pretrained_model("celeba_256a")
