
# TORCH 1.12, CUDA 11.3, TORCHVISION 0.13

import os
import sys
import time
from datetime import datetime, timedelta

from PIL import Image
import torch
from torch import nn, optim
from torch import distributed
from torch import multiprocessing

from torchvision import datasets
from torchvision import transforms
from torch.utils import data

class ConvNeuralNet(nn.Module):
    def __init__(self, device_0, device_1, device_2):
        super().__init__()

        self.device_0 = device_0
        self.device_1 = device_1
        self.device_2 = device_2

        self.transform_0 = nn.Sequential(
                # (N, 3, 128, 128)
                nn.Conv2d(in_channels=3, out_channels=512, padding=3, kernel_size=7),
                nn.BatchNorm2d(num_features=512, affine=False),
                nn.ELU(),

                # (N, 512, 128, 128)
                nn.MaxPool2d(kernel_size=2),

                # (N, 512, 64, 64)
                nn.Conv2d(in_channels=512, out_channels=256, padding=0, kernel_size=1),
                nn.BatchNorm2d(num_features=256, affine=False),
                nn.ELU(),

                # (N, 256, 64, 64)
                nn.Conv2d(in_channels=256, out_channels=1024, padding=2, kernel_size=5),
                nn.BatchNorm2d(num_features=1024, affine=False),
                nn.ELU(),

                # (N, 1024, 64, 64)
                nn.MaxPool2d(kernel_size=2),

                # (N, 1024, 32, 32)
                nn.Conv2d(in_channels=1024, out_channels=512, padding=0, kernel_size=1), 
                nn.BatchNorm2d(num_features=512, affine=False),
                nn.ELU(),

                # (N, 512, 32, 32)
                nn.Conv2d(in_channels=512, out_channels=2048, padding=2, kernel_size=5), 
                nn.BatchNorm2d(num_features=2048, affine=False),
                nn.ELU(),

                # (N, 2048, 32, 32)
                nn.Conv2d(in_channels=2048, out_channels=1024, padding=0, kernel_size=1),
                nn.BatchNorm2d(num_features=1024, affine=False),
                nn.ELU(),

                # (N, 1024, 32, 32)
                nn.Conv2d(in_channels=1024, out_channels=4096, padding=2, kernel_size=5),
                nn.BatchNorm2d(num_features=4096, affine=False),
                nn.ELU(),

                # (N, 4096, 32, 32)
                nn.MaxPool2d(kernel_size=2)
                ).to(device_0)


        self.transform_1 = nn.Sequential(
                # (N, 4096, 16, 16)
                nn.Conv2d(in_channels=4096, out_channels=2048, padding=0, kernel_size=1),
                nn.BatchNorm2d(num_features=2048, affine=False),
                nn.ELU(),

                # (N, 2048, 16, 16)
                nn.Conv2d(in_channels=2048, out_channels=4096, padding=2, kernel_size=5),
                nn.BatchNorm2d(num_features=4096, affine=False),
                nn.ELU(),

                # (N, 4096, 16, 16)
                nn.Conv2d(in_channels=4096, out_channels=2048, padding=0, kernel_size=1),
                nn.BatchNorm2d(num_features=2048, affine=False),
                nn.ELU(),

                # (N, 2048, 16, 16)
                nn.Conv2d(in_channels=2048, out_channels=8192, padding=2, kernel_size=5),
                nn.BatchNorm2d(num_features=8192, affine=False),
                nn.ELU(),

                # (N, 8192, 16, 16)
                nn.MaxPool2d(kernel_size=2),

                # (N, 8192, 8, 8)
                nn.Conv2d(in_channels=8192, out_channels=4096, padding=0, kernel_size=1),
                nn.BatchNorm2d(num_features=4096, affine=False),
                nn.ELU(),

                # (N, 4096, 8, 8)
                nn.Conv2d(in_channels=4096, out_channels=16384, padding=1, kernel_size=3),
                nn.BatchNorm2d(num_features=16384, affine=False),
                nn.ELU(),

                # (N, 16384, 8, 8)
                nn.MaxPool2d(kernel_size=2),

                # (M, 16384, 4, 4)
                nn.Flatten()

                ).to(device_1)

        self.transform_2 = nn.Sequential(
                # (N, 262144) 
                nn.Linear(in_features=262144, out_features=8192),
                nn.BatchNorm1d(num_features=8192, affine=False),
                nn.ELU(),

                # (N, 8192)
                nn.Linear(in_features=8192, out_features=8192),
                nn.BatchNorm1d(num_features=8192, affine=False),
                nn.ELU(),

                # (N, 8192)
                nn.Linear(in_features=8192, out_features=4096),
                nn.BatchNorm1d(num_features=4096, affine=False),
                nn.ELU(),

                # (N, 4096)
                nn.Linear(in_features=4096, out_features=4096),
                nn.BatchNorm1d(num_features=4096, affine=False),
                nn.ELU(),

                # (N, 4096)
                nn.Linear(in_features=4096, out_features=4),
                nn.BatchNorm1d(num_features=4, affine=False),
                nn.Softmax(dim=1)

                # (N, 4)
                ).to(device_2)


        nn.init.kaiming_uniform_(self.transform_0[0].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_0[4].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_0[7].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_0[11].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_0[14].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_0[17].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_0[20].weight, a=1e-4)

        nn.init.kaiming_uniform_(self.transform_1[0].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_1[3].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_1[6].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_1[9].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_1[13].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_1[16].weight, a=1e-4)

        nn.init.kaiming_uniform_(self.transform_2[0].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_2[3].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_2[6].weight, a=1e-4)
        nn.init.kaiming_uniform_(self.transform_2[9].weight, a=1e-4)
        nn.init.xavier_uniform_(self.transform_2[12].weight)

    def forward(self, images):
        transform_0_out = self.transform_0(images.to(self.device_0))
        transform_1_out = self.transform_1(transform_0_out.to(self.device_1))
        transform_2_out = self.transform_2(transform_1_out.to(self.device_2))
        return transform_2_out

def test(test_set_loadr, reduced_model, compute_error, epoch, output_gpu):
    reduced_model.eval()
    total_error = 0
    total_correct = 0
    
    with torch.no_grad():
        for images, labels in test_set_loadr:
            predictions = reduced_model(images)
            batch_error = compute_error(predictions, labels.to(output_gpu))
            total_error += batch_error.item()
            total_correct += (predictions.argmax(1) == labels.to(output_gpu)).type(torch.float32).sum().item()
            
    average_batch_error = total_error / len(test_set_loadr)
    accuracy = (total_correct / len(test_set_loadr.dataset)) * 100
    print(f"Epoch {epoch} @", datetime.now().strftime("%I:%M %p"), 
            f": {average_batch_error:.4f} C.E. Loss, {accuracy:.2f}% Accuracy", flush=True)

def train(valid_set_loadr, n_epoch):
    distributed.init_process_group(backend="nccl")

    # information about process 
    n_local_processes = 1
    n_gpus_per_process = 3
    global_process_id = distributed.get_rank()
    local_process_id = global_process_id % n_local_processes
    offset_to_gpu_set = local_process_id * n_gpus_per_process

    # dataset, dataloader, data augmentations 
    train_set = datasets.ImageFolder(root = os.getcwd() + "/data/train", transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize(mean=(0.487, 0.487, 0.487), std=(0.136, 0.136, 0.136))
    ]))
    train_set_samplr = data.distributed.DistributedSampler(dataset=train_set, shuffle=True, drop_last=True)
    train_set_loadr = data.DataLoader(dataset=train_set, batch_size=10, pin_memory=True, num_workers=8, sampler=train_set_samplr)

    # device configurations
    gpu_0 = torch.device("cuda", offset_to_gpu_set)
    gpu_1 = torch.device("cuda", offset_to_gpu_set + 1)
    gpu_2 = torch.device("cuda", offset_to_gpu_set + 2)

    # model, optimizer, l.r. scheduler, loss function
    model_parallel = ConvNeuralNet(gpu_0, gpu_1, gpu_2)
    sync_bn_model_parallel = nn.SyncBatchNorm.convert_sync_batchnorm(model_parallel)
    ddp_model = nn.parallel.DistributedDataParallel(sync_bn_model_parallel)

    optimizer = optim.SGD(ddp_model.parameters(), lr=3e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    compute_error = nn.CrossEntropyLoss()
    start_epoch = 0

    # load checkpoint
    checkpoint_file = os.getcwd() + "/model/checkpoint.pt"
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
        ddp_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if global_process_id == 0:
            print(f"Load Checkpoint @", datetime.now().strftime("%I:%M %p"), flush=True)

    # train
    for epoch in range(start_epoch, n_epoch):
        ddp_model.train()
        train_set_samplr.set_epoch(epoch)

        for images, labels in train_set_loadr:
            predictions = ddp_model(images)
            batch_error = compute_error(predictions, labels.to(gpu_2))
            optimizer.zero_grad()
            batch_error.backward()
            optimizer.step()

        scheduler.step()

        distributed.barrier()
        if global_process_id == 0:
            test(valid_set_loadr, ddp_model, compute_error, epoch, gpu_2)
            
            # save checkpoint
            if (epoch + 1) % 32 == 0:
                checkpoint = {
                        "model_state_dict": ddp_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch
                        }
                torch.save(checkpoint, checkpoint_file)
                print(f"Save Checkpoint @", datetime.now().strftime("%I:%M %p"), flush=True)
        distributed.barrier()

    distributed.destroy_process_group()

if __name__  == "__main__":
    valid_set = datasets.ImageFolder(root = os.getcwd() + "/data/valid", transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize(mean=(0.487, 0.487, 0.487), std=(0.136, 0.136, 0.136))
    ]))
    valid_set_loadr = data.DataLoader(valid_set, batch_size=128, drop_last=False, shuffle=False, pin_memory=True)
    train(valid_set_loadr, 256)

