import os
import sys

from preprocess import get_sampled_data
from model import get_model, device
from model_parts import MelDataset, EarlyStopDetector

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm


def train(model, train_loader, val_loader,
          epochs=50, lr=3e-4, early_stop=5, save_path=os.path.join('model', 'model.ptm')):

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True, betas=(0.9, 0.999))
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    early_stop_detector = EarlyStopDetector(max_steps=early_stop, reverse=True)

    train_LOSS = []
    val_LOSS = []

    for epoch in range(epochs):

        if early_stop_detector.check_for_stop():
            break

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
            running_loss = 0.

            for idx, (x, y) in loop:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                # Forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    y_ = model(x)
                    loss = criterion(y_, x - y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.sum().item()

                    loop.set_description(f'Epoch [{epoch+1}/{epochs}] {phase}')
                    loop.set_postfix(loss=running_loss/(idx+1))

            epoch_loss = running_loss / len(dataloader)

            if phase == 'train':
                train_LOSS.append(epoch_loss)
                sheduler.step()
            else:
                val_LOSS.append(epoch_loss)
                if early_stop_detector.check_for_best_score(epoch_loss):
                    # Save best score model weights
                    torch.save(model.state_dict(), save_path)

    return model, (train_LOSS, val_LOSS)


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    # Parsing arguments
    # First argument must be path to data dir, otherwise use default dir
    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = os.path.join(os.getcwd(), 'data')

    # Path to save trained model
    net_path = os.path.join('model', 'model.ptm')

    # Define dirs
    train_data_clean_dir = os.path.join(data_path, 'train', 'clean')
    train_data_noisy_dir = os.path.join(data_path, 'train', 'noisy')
    val_data_clean_dir = os.path.join(data_path, 'val', 'clean')
    val_data_noisy_dir = os.path.join(data_path, 'val', 'noisy')

    # Get 80x80 samples from data
    train_data_clean = get_sampled_data(train_data_clean_dir, n_loads=50)
    train_data_noisy = get_sampled_data(train_data_noisy_dir, n_loads=50)
    val_data_clean = get_sampled_data(val_data_clean_dir, n_loads=50)
    val_data_noisy = get_sampled_data(val_data_noisy_dir, n_loads=50)

    # Make train and val datasets
    train_dataset = MelDataset(list(zip(train_data_noisy, train_data_clean)), transform=True)
    val_dataset = MelDataset(list(zip(val_data_noisy, val_data_clean)), transform=True)

    # Make loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, pin_memory=True)

    print('Start training...')

    # Get model
    model = get_model()

    # Train model
    model, (train_loss, val_loss) = train(model, train_loader, val_loader,
                                          epochs=20, lr=3e-3, early_stop=3, save_path=net_path)

    print('Model trained successfully')

    plt.plot(train_loss, 'r', label='train')
    plt.plot(val_loss, 'g', label='val')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss (Smooth L1)')
    plt.show()
