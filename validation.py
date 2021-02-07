import sys
import os
import torch
from model import get_model
from preprocess import get_raw_data
from tqdm import tqdm

# Parsing arguments
# First argument must be path to validation data dir, otherwise use default dir
try:
      val_data_path = sys.argv[1]
except IndexError:
      val_data_path = os.path.join('data', 'val')

# Second argument - path to pretrained model file,
# otherwise use default
try:
    net_path = sys.argv[2]
except IndexError:
    net_path = os.path.join('model', 'model.ptm')

# Dirs with validation data
val_data_clean_dir = os.path.join(val_data_path, 'clean')
val_data_noisy_dir = os.path.join(val_data_path, 'noisy')

# Get files from validation data
val_data_clean = get_raw_data(val_data_clean_dir)
val_data_noisy = get_raw_data(val_data_noisy_dir)

# Take model
model = get_model()
# Load pretrained model
if os.path.isfile(net_path):
    model.load_state_dict(torch.load(net_path))
else:
    raise BaseException('Neural Network not fitted!')

model.eval()

# Use MSE criterion to evaluate
criterion = torch.nn.MSELoss()

# Evaluate
with torch.no_grad():
    # Total loss variables
    val_loss, noisy_loss = 0., 0.

    loop = tqdm(enumerate(zip(val_data_noisy, val_data_clean)), total=len(val_data_noisy), leave=True)
    # Validation loop
    for batch_idx, (x, y) in loop:
        y = torch.from_numpy(y)

        # Predict and calculate MSE
        y_ = model.predict(x)

        loss = criterion(y_, y)
        val_loss += torch.sum(loss).item()

        # Calculate MSE of noisy data
        loss = criterion(torch.from_numpy(x), y)
        noisy_loss += torch.sum(loss).item()

        # Update progress bar
        loop.set_description(f'Evaluating')
        loop.set_postfix(noisy_mse=noisy_loss/(batch_idx+1), denoised_mse=val_loss/(batch_idx+1))

print(f'Noisy data MSE: summary - {noisy_loss:.2f}, mean - {noisy_loss/len(val_data_noisy):.2f}')
print(f'Denoised data MSE: summary - {val_loss:.2f}, mean - {val_loss/len(val_data_noisy):.2f}')
