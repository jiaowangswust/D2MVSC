import torch.nn as nn
import torch
from se import SE_block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1' 
class Networks(nn.Module):
    def __init__(self):
        super(Networks, self).__init__()

        #view1
        self.encoder11 = nn.Sequential(
            nn.Linear(216, 256, bias=True),
            nn.ReLU(),
        )
        self.encoder12 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
        )
        self.decoder11 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
        )
        self.decoder12 = nn.Sequential(
            nn.Linear(256, 216, bias=True),
            nn.ReLU(),
        )
        #view2
        self.encoder21 = nn.Sequential(
            nn.Linear(76, 256, bias=True),
            nn.ReLU(),
        )
        self.encoder22 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
        )
        self.decoder21 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
        )
        self.decoder22 = nn.Sequential(
            nn.Linear(256, 76, bias=True),
            nn.ReLU(),
        )
        #view3
        self.encoder31 = nn.Sequential(
            nn.Linear(64, 256, bias=True),
            nn.ReLU(),
        )
        self.encoder32 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
        )
        self.decoder31 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
        )
        self.decoder32 = nn.Sequential(
            nn.Linear(256, 64, bias=True),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(nn.ConvTranspose2d(3, 1, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU())
        self.weight = nn.Parameter(1.0e-4 * torch.ones(2000, 2000))
        self.weight1 = nn.Parameter(1.0e-4 * torch.ones(2000, 2000))
        self.weight2 = nn.Parameter(1.0e-4 * torch.ones(2000, 2000))
        self.weight3 = nn.Parameter(1.0e-4 * torch.ones(2000, 2000))

    
    def forward(self, input1, input2, input3):
        output1 = self.encoder11(input1)
        output1 = self.encoder12(output1)
        output1 = self.decoder11(output1)
        output1 = self.decoder12(output1)

        output2 = self.encoder21(input2)
        output2 = self.encoder22(output2)
        output2 = self.decoder21(output2)
        output2 = self.decoder22(output2)

        output3 = self.encoder31(input3)
        output3 = self.encoder32(output3)
        output3 = self.decoder31(output3)
        output3 = self.decoder32(output3)

        return output1, output2, output3
    def forward2(self, input1, input2, input3):
        coef1 = self.weight1 - torch.diag(torch.diag(self.weight1))  
        coef2 = self.weight2 - torch.diag(torch.diag(self.weight2))  
        coef3 = self.weight3 - torch.diag(torch.diag(self.weight3))  

        # view1
        z11 = self.encoder11(input1)
        zcoef11 = torch.matmul(coef1, z11)
        z12 = self.encoder12(z11)
        zcoef12 = torch.matmul(coef1, z12)

        zoutput11 = self.decoder11(zcoef12)
        zoutput12 = self.decoder12(zoutput11)
        zout12 = self.decoder12(zcoef11)

        # view2
        z21 = self.encoder21(input2)
        zcoef21 = torch.matmul(coef2, z21)
        z22 = self.encoder22(z21)
        zcoef22 = torch.matmul(coef2, z22)

        zoutput21 = self.decoder21(zcoef22)
        zoutput22 = self.decoder22(zoutput21)
        zout22 = self.decoder22(zcoef21)

        # view3
        z31 = self.encoder31(input3)
        zcoef31 = torch.matmul(coef3, z31)
        z32 = self.encoder32(z31)
        zcoef32 = torch.matmul(coef3, z32)

        zoutput31 = self.decoder31(zcoef32)
        zoutput32 = self.decoder32(zoutput31)
        zout32 = self.decoder32(zcoef31)


        coeff = torch.stack((coef1, coef2, coef3), dim=0)
        coeff = torch.reshape(coeff, [-1, 3, 2000, 2000])
        semodel = SE_block(3, 3)
        semodel = semodel.to(device)
        coeff = semodel(coeff)
        c1, c2, c3 = torch.chunk(coeff, 3, dim=1)
        c = self.fusion(coeff)
        coef = c.view(2000, 2000)

        return z11, z12, zcoef11, zcoef12, zoutput12, zout12, coef1, coef2, coef3, coef, z21, z22, zcoef21, zcoef22, zoutput22, zout22,z31, z32, zcoef31, zcoef32, zoutput32, zout32


    def forward3(self, input1, input2, input3):
        coef1 = self.weight1 - torch.diag(torch.diag(self.weight1))  
        coef2 = self.weight2 - torch.diag(torch.diag(self.weight2))  
        coef3 = self.weight3 - torch.diag(torch.diag(self.weight3))  

        # view1
        z11 = self.encoder11(input1)
        zcoef11 = torch.matmul(coef1, z11)
        z12 = self.encoder12(z11)
        zcoef12 = torch.matmul(coef1, z12)

        zoutput11 = self.decoder11(zcoef12)
        zoutput12 = self.decoder12(zoutput11)
        zout12 = self.decoder12(zcoef11)

        # view2
        z21 = self.encoder21(input2)
        zcoef21 = torch.matmul(coef2, z21)
        z22 = self.encoder22(z21)
        zcoef22 = torch.matmul(coef2, z22)

        zoutput21 = self.decoder21(zcoef22)
        zoutput22 = self.decoder22(zoutput21)
        zout22 = self.decoder22(zcoef21)

        # view3
        z31 = self.encoder31(input3)
        zcoef31 = torch.matmul(coef3, z31)
        z32 = self.encoder32(z31)
        zcoef32 = torch.matmul(coef3, z32)

        zoutput31 = self.decoder31(zcoef32)
        zoutput32 = self.decoder32(zoutput31)
        zout32 = self.decoder32(zcoef31)


        coeff = torch.stack((coef1, coef2, coef3), dim=0)
        coeff = torch.reshape(coeff, [-1, 3, 2000, 2000])
        semodel = SE_block(3, 3)
        semodel = semodel.to(device)
        coeff = semodel(coeff)
        c1, c2, c3 = torch.chunk(coeff, 3, dim=1)
        c = self.fusion(coeff)
        coef = c.view(2000, 2000)
        
        return z11, z12, zcoef11, zcoef12, zoutput12, zout12, coef1, coef2, coef3, coef, z21, z22, zcoef21, zcoef22, zoutput22, zout22,z31, z32, zcoef31, zcoef32, zoutput32, zout32






