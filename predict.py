import os
import sys
import numpy as np
from torch import load, no_grad
from model import get_model

# First argument - file to denoise
try:
    file_path = sys.argv[1]
except IndexError:
    raise BaseException('No file to denoise!')

# Second argument - path to save denoised file,
# otherwise use default
try:
    savefile_path = sys.argv[2]
except IndexError:
    savefile_path = 'predicted.npy'

# Third argument - path to pretrained model file,
# otherwise use default
try:
    net_path = sys.argv[3]
except IndexError:
    net_path = os.path.join('model', 'model.ptm')

# Take model
model = get_model()

# Load pretrained model
if os.path.isfile(net_path):
    model.load_state_dict(load(net_path))
else:
    raise BaseException('Neural Network not fitted!')

model.eval()

# Load data from file and convert to torch.tensor
mel_data = np.load(file_path).astype(np.single)

# Denoise
with no_grad():
    denoised_mel_data = model.predict(mel_data).numpy()

# Save denoised data to file
np.save(savefile_path, denoised_mel_data)
