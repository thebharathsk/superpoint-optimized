import timm
import torch
import torch.nn as nn

class SuperPointOptimized(nn.Module):
    def __init__(self):
        super(SuperPointOptimized, self).__init__()
        
        # Backbone
        backbone = timm.create_model('mobilenetv2_140', pretrained=True)
        layers = list(backbone.children())

        layers_new = []
        layers_new.extend(layers[0:2])
        layers_new.extend(list(layers[2]))

        self.encoder_truncated = torch.nn.Sequential(*layers_new[0:5])

        # Variables
        c4 = 48
        c5 = 256
        d1 = 256
        det_h = 65
        
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        
        # ReLU
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self,x):
        # Extract features
        fts = self.encoder_truncated(x)
        
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(fts)))
        semi = self.bnPb(self.convPb(cPa))
        
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(fts)))
        desc = self.bnDb(self.convDb(cDa))

        # Normalize descriptors
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.

        return semi, desc