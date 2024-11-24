from comet_ml import Experiment, ExistingExperiment

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path
import pickle as pkl

import torch
import torch.nn as nn
from monai.networks.nets import PatchDiscriminator
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter  # do not need this

from scripts.transforms import VAE_Transform
from scripts.utils import KL_loss, define_instance, dynamic_infer
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image

# additional imports
from pvg.runner.utilities.io import IO

from dataset import DatasetDescription, CustomDataLoader
from utils import build_batch

import numpy as np
import matplotlib.pyplot as plt
import ignite.distributed as idist
import argparse
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

DEVICE = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='Parent directory of LORIS dataset (SLURM_TMPDIR)')
parser.add_argument('--exp', required=True, help='Experiment number, for labelling')
parser.add_argument('--debug', action='store_true', required=False, help='Whether to deactivate certain aspects of the script for debugging (e.g., Comet logging)')
parser.add_argument('--device-ids', nargs='+', type=int, required=False, help='Device IDs for Data Parallel')
pargs = parser.parse_args()

data_dir = pargs.data_dir
exp = pargs.exp
debug = pargs.debug
device_ids = [f'cuda:{gpu}' for gpu in pargs.device_ids]

# --------------------
# Setup data directory
# --------------------
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory


# ------------------------
# Instantiate data loaders
# ------------------------
splits_path = '/home/ritchiey/projects/def-arbeltal/ritchiey/capstone/old_splits.pickle'  # using 1x1x3_v4_0_w000 for now...
rootdir_dict = {"MRI_AND_LABEL": f"{data_dir}/loris_1x1x3_v4_0_w000"}
splits = IO.load_pickle(splits_path)
dataset_descriptor = DatasetDescription()
dataset_descriptor.setup_datasets(splits["experiment"], rootdir_dict)

dataloader = CustomDataLoader()
dataloader_train = dataloader.get_train_loader(dataset_descriptor.train_dataset)
dataloader_val = dataloader.get_val_loader(dataset_descriptor.val_dataset)
dataloader_test = dataloader.get_test_loader(dataset_descriptor.test_dataset)


# --------------------------------
# Visualize intensity distribution
# --------------------------------
img = next(iter(dataloader_train))
img = build_batch(img, DEVICE)['MRI']

fig, ax = plt.subplots()
ax.hist(img.detach().cpu().numpy().flatten(), bins=30, log=True)

fig.savefig('intensity.png', dpi=300)


# ----------------------------
# Read in environment settings
# ----------------------------
args = argparse.Namespace()

environment_file = "./configs/environment_maisi_vae_train.json"
env_dict = json.load(open(environment_file, "r"))
for k, v in env_dict.items():
    setattr(args, k, v)
    print(f"{k}: {v}")

# model path
Path(args.model_dir).mkdir(parents=True, exist_ok=True)
trained_g_path = os.path.join(args.model_dir, f"autoencoder_{exp}.pt")
trained_d_path = os.path.join(args.model_dir, f"discriminator_{exp}.pt")
print(f"Trained model will be saved as {trained_g_path} and {trained_d_path}.")


# ------------------------------
# Read in configuration settings
# ------------------------------
config_file = "./configs/config_maisi.json"  # defines autoencoder architecture
config_dict = json.load(open(config_file, "r"))
for k, v in config_dict.items():
    setattr(args, k, v)

# check the format of inference inputs
config_train_file = "./configs/config_maisi_vae_train.json"
config_train_dict = json.load(open(config_train_file, "r"))
for k, v in config_train_dict["data_option"].items():
    setattr(args, k, v)
    print(f"{k}: {v}")
for k, v in config_train_dict["autoencoder_train"].items():
    setattr(args, k, v)
    print(f"{k}: {v}")

print("Network definition and training hyperparameters have been loaded.")


# -------------------
# Setup Comet logging
# -------------------
if not(debug):
    comet_experiment = Experiment(api_key='dSduxIFjB8xooDgnz3sq39eZc',
                                  project_name='maisi_finetuning',
                                  workspace='capstone2024',
                                  auto_metric_logging=False,
                                  display_summary_level=0)

    comet_experiment.set_name(f'maisi{exp}')
    comet_experiment.log_parameters(vars(args))


# -----------------------------------
# Set determinism for reproducibility
# -----------------------------------
set_determinism(seed=0)


# -------------------
# Initialize networks
# -------------------
args.autoencoder_def["num_splits"] = 1

# load pre-trained autoencoder, which we will finetune
state_dict = torch.load('models/autoencoder_epoch273.pt', map_location='cpu')

if not(device_ids is None):
    autoencoder = define_instance(args, "autoencoder_def")
    autoencoder = nn.DataParallel(autoencoder, device_ids=device_ids).cuda()
    autoencoder.module.load_state_dict(state_dict)

    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    )
    discriminator = nn.DataParallel(discriminator, device_ids=device_ids).cuda()

else:
    autoencoder = define_instance(args, "autoencoder_def")
    autoencoder.load_state_dict(state_dict).to(DEVICE)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(DEVICE)


# ----------------------
# Training configuration
# ----------------------
# config loss and loss weight
if args.recon_loss == "l2":
    intensity_loss = MSELoss()
    print("Use l2 loss")
else:
    intensity_loss = L1Loss(reduction="mean")
    print("Use l1 loss")
adv_loss = PatchAdversarialLoss(criterion="least_squares")

pl_path = '/home/ritchiey/projects/def-arbeltal/ritchiey/capstone/perceptual_loss.pth'
pl = torch.load(pl_path, weights_only=False).eval().to(DEVICE)
loss_perceptual = (pl)

# config optimizer and lr scheduler
if not(device_ids is None):
    optimizer_g = torch.optim.Adam(params=autoencoder.module.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator.module.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
else:
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)

# TODO: please adjust the learning rate warmup rule based on your dataset and n_epochs
def warmup_rule(epoch):
    return 1.0

    # # learning rate warmup rule
    # if epoch < 10:
    #     return 0.01
    # elif epoch < 20:
    #     return 0.1
    # else:
    #     return 1.0

scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

# set AMP scaler
if args.amp:
    # test use mean reduction for everything
    scaler_g = GradScaler(DEVICE, init_scale=2.0**8, growth_factor=1.5)
    scaler_d = GradScaler(DEVICE, init_scale=2.0**8, growth_factor=1.5)


# --------
# Training
# --------

# Initialize variables
val_interval = args.val_interval
best_val_recon_epoch_loss = 10000000.0
total_step = 0
start_epoch = 0
max_epochs = args.n_epochs

# Setup validation inferer
val_inferer = (
    SlidingWindowInferer(
        roi_size=args.val_sliding_window_patch_size,
        sw_batch_size=1,
        progress=False,
        overlap=0.0,
        device=torch.device("cpu"),
        sw_device=DEVICE,
    )
    if args.val_sliding_window_patch_size
    else SimpleInferer()
)

def loss_weighted_sum(losses):
    return losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + args.perceptual_weight * losses["p_loss"]

# Training and validation loops
for epoch in range(start_epoch, max_epochs):
    print("lr:", scheduler_g.get_lr())
    autoencoder.train()
    discriminator.train()
    train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0, "g_loss": 0, "d_loss": 0}

    for batch_idx, batch in tqdm(enumerate(dataloader_train)):

        batch = build_batch(batch, DEVICE)  # key step
        images = batch['MRI'].contiguous()

        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=args.amp):
            
            # Train Generator
            reconstruction, z_mu, z_sigma = autoencoder(images)
            losses = {
                "recons_loss": intensity_loss(reconstruction, images),
                "kl_loss": KL_loss(z_mu, z_sigma),
                "p_loss": loss_perceptual(reconstruction.float(), images.float()),
            }
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = loss_weighted_sum(losses) + args.adv_weight * generator_loss

            if args.amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            # Train Discriminator
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            if args.amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

            losses['g_loss'] = loss_g.detach().cpu()
            losses['d_loss'] = loss_d.detach().cpu()

        # Log sample reconstruction (since training proceeds so slowly, sub-sample an epoch)
        if batch_idx % 100 == 0 and not(debug):
            for i in range(reconstruction.shape[0]):
                fig, ax = plt.subplots()
                fig.dpi = 500
                img_slice = reconstruction[i, 0, 32, :, :].detach().cpu().numpy()
                ax.imshow(img_slice, cmap='gray')
                comet_experiment.log_figure(f"Train {i}/{reconstruction.shape[0]}", fig)

        # Log training loss
        total_step += 1
        for loss_name, loss_value in losses.items():
            train_epoch_losses[loss_name] += loss_value.item()
    
    # Step LR schedular
    scheduler_g.step()
    scheduler_d.step()

    # Compute average validation loss
    for key in train_epoch_losses:
        train_epoch_losses[key] /= len(dataloader_train)

    # Log losses to Comet
    if not(debug):
        comet_experiment.log_metrics(train_epoch_losses)

    print(f"Epoch {epoch} train_vae_loss {loss_weighted_sum(train_epoch_losses)}: {train_epoch_losses}.")
    torch.save(autoencoder.state_dict(), trained_g_path)
    torch.save(discriminator.state_dict(), trained_d_path)
    print("Save trained autoencoder to", trained_g_path)
    print("Save trained discriminator to", trained_d_path)

    # Validation
    if epoch % val_interval == 0:
        autoencoder.eval()
        val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
        val_loader_iter = iter(dataloader_val)
        for batch_idx, batch in tqdm(enumerate(dataloader_val)):
            with torch.no_grad():
                with autocast("cuda", enabled=args.amp):
                    
                    batch = build_batch(batch, 'cpu')  # key step
                    images = batch["MRI"]

                    reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, images)
                    reconstruction = reconstruction.to(DEVICE)
                    val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, images.to(DEVICE)).item()
                    val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                    val_epoch_losses["p_loss"] += loss_perceptual(reconstruction, images.to(DEVICE)).item()
            
            # Log sample reconstruction (since validation proceeds so slowly, sub-sample an epoch)
            if batch_idx % 100 == 0 and not(debug):
                for i in range(reconstruction.shape[0]):
                    fig, ax = plt.subplots()
                    fig.dpi = 500
                    img_slice = reconstruction[i, 0, 32, :, :].detach().cpu().numpy()
                    ax.imshow(img_slice, cmap='gray')
                    comet_experiment.log_figure(f"Validation {i}/{reconstruction.shape[0]}", fig)

        # Compute average validation loss
        for key in val_epoch_losses:
            val_epoch_losses[key] /= len(dataloader_val)

        # Log losses
        if not(debug):
            comet_experiment.log_metrics(val_epoch_losses)

        val_loss_g = loss_weighted_sum(val_epoch_losses)
        print(f"Epoch {epoch} val_vae_loss {val_loss_g}: {val_epoch_losses}.")

        if val_loss_g < best_val_recon_epoch_loss:
            best_val_recon_epoch_loss = val_loss_g
            trained_g_path_epoch = f"{trained_g_path[:-3]}_epoch{epoch}.pt"
            torch.save(autoencoder.state_dict(), trained_g_path_epoch)
            print("Got best val vae loss.")
            print("Save trained autoencoder to", trained_g_path_epoch)

        # Monitor scale_factor
        # We'd like to tune kl_weights in order to make scale_factor close to 1.
        scale_factor_sample = 1.0 / z_mu.flatten().std()

        # Monitor reconstruction result
        center_loc_axis = find_label_center_loc(images[0, 0, ...])
        vis_image = get_xyz_plot(images[0, ...], center_loc_axis, mask_bool=False)
        vis_recon_image = get_xyz_plot(reconstruction[0, ...], center_loc_axis, mask_bool=False)

        show_image(vis_image, title="val image")
        show_image(vis_recon_image, title="val recon result")
