import torch
from model import SuperPointOptimized


if __name__ == "__main__":
    #load model
    model = SuperPointOptimized().to('cpu')

    #load checkpoint
    ckpt = torch.load('./superpoint_opt.pth', map_location='cpu')

    #load weights
    model.load_state_dict(ckpt)