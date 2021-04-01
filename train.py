import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py

from dataset import PaviaDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from model import ThreeDFCNN
import torch.nn.functional as F

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.00005
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():

    #加载数据
    dataset = PaviaDataset(file_dir=args.input_data)
    #print(type(dataset[0]))
    #print(dataset[0].shape)
    #print(len(dataset))
    #x, y = dataset[0]
    #print(type(x))
    #print(type(y))
    #print(x.shape)
    #print(y.shape)

    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    #image, label = next(iter(dataloader))
    #print(type(image), type(label))
    #print(image.shape, label.shape)

    model = ThreeDFCNN()
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_epoch = args.n_epoch

    for epoch in range(n_epoch):
        model.train()
        for batch_idx, (image, label) in enumerate(dataloader):
            image = image.permute(0, 4, 3, 1, 2)
            label = label.permute(0, 4, 3, 1, 2)
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            print(image.shape)
            print(label.shape)
            pred = model(image)

            cost = F.mse_loss(label, pred)
            optimizer.zero_grad()

            cost.backward()
            optimizer.step()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--save',
                        default='./save',
                        dest='save',
                        type=str,
                        nargs=1,
                        help="Path to save the checkpoints to")
    parser.add_argument('-D', '--data',
                        default='data_process/pa_train_3d_all_data.h5',
                        dest='input_data',
                        type=str,
                        nargs=1,
                        help="Training data directory")
    parser.add_argument('-E', '--epoch',
                        default=1,
                        dest='n_epoch',
                        type=int,
                        nargs=1,
                        help="Training epochs must be a multiple of 5")
    args = parser.parse_args()
    print(args)
    train()
