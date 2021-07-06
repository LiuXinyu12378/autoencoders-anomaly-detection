from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from ignite.metrics import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Grayscale
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pathlib
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from torch.optim import Adam
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
import os
from model import AnomalyAE
import cv2
import numpy as np
import matplotlib.pyplot as plt

path_base = os.path.dirname(__file__)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def create_datagen(data_dir, batch_size=1):
    transform = Compose([Grayscale(),ToTensor()])
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)
    return dataloader

optimizer = Adam
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = F.mse_loss
model = AnomalyAE()

def interface(model, optimizer,loss_fn,
           device,load_weight_path="./tensorboard_logs_05112020_05-08/models/best_model0_5_loss=-0.0001.pt", save_graph=False):
    """Training logic for the wavelet model

    Arguments:
        model {pytorch model}       -- the model to be trained
        optimizer {torch optim}     -- optimiser to be used
        loss_fn                        -- loss_fn function
        train_loader {dataloader}   -- training dataloader
        val_loader {dataloader}     -- validation dataloader
        log_dir {str}               -- the log directory
        device {torch.device}       -- the device to be used e.g. cpu or cuda
        epochs {int}                -- the number of epochs
        log_interval {int}          -- the log interval for train batch loss

    Keyword Arguments:
        load_weight_path {str} -- Model weight path to be loaded (default: {None})
        save_graph {bool}      -- whether to save the model graph (default: {False})

    Returns:
        None
    """
    model.to(device)
    if load_weight_path is not None:
        model.load_state_dict(torch.load(load_weight_path,map_location=torch.device(device)))

    optimizer = optimizer(model.parameters())


    model.eval()
    print(path_base)
    data_path = path_base+"/test_image/"
    batch = create_datagen(data_path)
    # img_gray = cv2.imread("image/0045.PNG")
    # img_gray = cv2.cvtColor(img_gray,cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.resize(img_gray,(512,512))
    # print(img_gray.shape)
    # tensor_data = torch.FloatTensor(img_gray)
    # tensor_data = tensor_data.unsqueeze(0)
    # tensor_data = tensor_data.unsqueeze(0).div(255.)
    # np.set_printoptions(threshold=np.inf)
    with torch.no_grad():
        for   (x,_)  in batch:
            print(x)
        # x = tensor_data
        # x = x.to(device)
            y = model(x)
            x_ = x.squeeze(0).squeeze(0).mul(255.).numpy().astype(np.int8)
            y_ = y.squeeze(0).squeeze(0).mul(255.).numpy().astype(np.int8)
            x_y = x_ - y_
            x_y2 = np.maximum(x_y, -x_y)
        # print(x_)
        # print(y_)
        # print(x_y)
        # print(x_y2)

        # idx1 = np.argmax(x_y2, axis=0)
        # idx2 = np.argmax(x_y2, axis=1)
        # print(x_y2.shape)
        # print(idx1,idx2)
            print(np.max(x_y2))
        # index = np.unravel_index(x_y2.argmax(), x_y2.shape)
        # print(index)
        # print(x_y2[300:400,0:100])
            cv2.imshow("x",x_)
            cv2.imshow("y",y_)
        # plt.grid()
        # plt.imshow(y_)
            cv2.imshow("x-y",x_y2)
            cv2.waitKey(0)
            # plt.show()
            loss = loss_fn(y,x)
        return "{:.10f}".format(loss.item())


    # evaluator = Engine(evaluate_function)
    #
    # RunningAverage(output_transform=lambda x:x).attach(evaluator,'loss')


if __name__ == '__main__':
    print(interface(model,optimizer,loss_fn,device))




