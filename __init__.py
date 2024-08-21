import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision.io import read_image, ImageReadMode
from multilingual_clip import Config_MCLIP
import open_clip
import json
import pandas as pd
import random
from pathlib import Path
import cv2
import numpy as np
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForMaskedLM
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import os
import gc
import time
import math
from normalizer import normalize

