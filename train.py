import torch
from dataloader_uci import data_loader_train
from network import Networks
import metrics as metrics
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from se import SE_block
import torch.nn as nn
from munkres import Munkres
from torch.nn import functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1' 

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got 
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def graph_loss(Z, S):
    S = 0.5 * (S.permute(1, 0) + S)
    D = torch.diag(torch.sum(S, 1))
    L = D - S
    return 2 * torch.trace(torch.matmul(torch.matmul(Z.permute(1, 0), L), Z))


data_0 = sio.loadmat('uci-digit_m.mat')
data_dict = dict(data_0)
data0 = data_dict['truth']#.T
label_true = np.zeros(2000)
for i in range(2000):
    label_true[i] = data0[i]


r1 = 10#best
r2 = 0.01#best
learning_rate = 0.001# acc=98.75

model = Networks()
model = model.to(device)
model.load_state_dict(torch.load('./AE1.pth'), strict=False)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0)
n_epochs = 1000
for epoch in range(n_epochs):
    for data in data_loader_train:
        train_imga, train_imgb, train_imgc = data
        input1 = train_imga.view(2000, 216)
        input1 = input1.to(device)
        input2 = train_imgb.view(2000, 76)
        input2 = input2.to(device)
        input3 = train_imgc.view(2000, 64)
        input3 = input3.to(device)
        output1, output2, output3 = model(input1, input2, input3)
        loss = criterion(output1, input1) +  criterion(output2, input2) +  criterion(output3, input3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 500 == 0:
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("Loss is:{:.4f}".format(loss.item()))
torch.save(model.state_dict(), './AE1.pth')

print("step2")
print("---------------------------------------")
criterion2 = torch.nn.MSELoss(reduction='sum')
optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0008, weight_decay=0.05)
n_epochs2 = 30
zero = torch.zeros(2000, 2000, requires_grad=True)
zero = zero.to(device)
for epoch in range(n_epochs2):
    for data in data_loader_train:
        train_imga, train_imgb, train_imgc = data
        input1 = train_imga.view(2000, 216)
        input1 = input1.to(device)
        input2 = train_imgb.view(2000, 76)
        input2 = input2.to(device)
        input3 = train_imgc.view(2000, 64)
        input3 = input3.to(device)
        z11, z12, zcoef11, zcoef12, zoutput12, zout12, coef1, coef2, coef3, coef, z21, z22, zcoef21, zcoef22, zoutput22, zout22,z31, z32, zcoef31, zcoef32, zoutput32, zout32 = model.forward2(input1, input2, input3)

        loss_re = criterion2(coef1, zero) + criterion2(coef2, zero) + criterion2(coef3, zero)
        loss_rc = criterion2(coef, zero)
        loss_e = criterion2(zcoef11, z11) + criterion2(zcoef12, z12) + criterion2(zcoef21, z21) + criterion2(zcoef22, z22) + criterion2(zcoef31, z31) + criterion2(zcoef32, z32) # + 0.5*criterion2(zcoef2, z2) +  0.5*criterion2(zcoef1, z1)
        loss_r = criterion2(zoutput12, input1) + criterion2(zout12, input1) + criterion2(zoutput22, input2) + criterion2(zout22, input2) +  criterion2(zoutput32, input3) + criterion2(zout32, input3)
        loss_g4 = graph_loss(z12, coef) + graph_loss(z11, coef)
        loss_g5 = graph_loss(z22, coef) + graph_loss(z21, coef)
        loss_g6 = graph_loss(z32, coef) + graph_loss(z31, coef)
        loss_g = loss_g4+loss_g5+loss_g6

        loss = loss_r + r1*loss_re + r1*loss_e + r2*loss_g + r1*loss_rc 
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

coef = torch.abs(coef)
commonZ = coef.cpu().detach().numpy()
alpha = max(0.4 - (10 - 1) / 10 * 0.1, 0.1)
commonZ = thrC(commonZ, alpha)
preds, _ = post_proC(commonZ, 10)
label_preds = best_map(label_true, preds)

sio.savemat('label_predsnew' +'.mat', {'label_preds': label_preds}) 


print("step3")
print("---------------------------------------")
optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=0.05)
n_epochs2 = 61
zero = torch.zeros(2000, 2000, requires_grad=True)
zero = zero.to(device)
ACC_FM = np.zeros((1, 51))
NMI_FM = np.zeros((1, 51))
FSCORE_FM = np.zeros((1, 51))
ARI_FM = np.zeros((1, 51))
PRECISION_FM = np.zeros((1, 51))
RECALL_FM = np.zeros((1, 51))

label_0 = sio.loadmat('label_predsnew.mat')
label_dict = dict(label_0)
label0 = label_dict['label_preds'].T
label_preds = np.zeros(2000)
for i in range(2000):
    label_preds[i] = label0[i]
print(label_preds)

label_preds = torch.LongTensor(label_preds)
label_preds = label_preds.to(device)
label_preds = label_preds - torch.ones_like(label_preds)

Q = F.one_hot(label_preds, num_classes=10)
Q = Q.to(dtype=torch.float)
Q = Q.to(device)

for epoch in range(n_epochs2):
    for data in data_loader_train:
        train_imga, train_imgb, train_imgc = data

        input1 = train_imga.view(2000, 216)
        input1 = input1.to(device)
        input2 = train_imgb.view(2000, 76)
        input2 = input2.to(device)
        input3 = train_imgc.view(2000, 64)
        input3 = input3.to(device)
 
        z11, z12, zcoef11, zcoef12, zoutput12, zout12, coef1, coef2, coef3, coef, z21, z22, zcoef21, zcoef22, zoutput22, zout22,z31, z32, zcoef31, zcoef32, zoutput32, zout32 = model.forward3(input1, input2, input3)

        loss_re = criterion2(coef1, zero) + criterion2(coef2, zero) + criterion2(coef3, zero)
        loss_rc = criterion2(coef, zero)
        loss_e = criterion2(zcoef11, z11) + criterion2(zcoef12, z12) + criterion2(zcoef21, z21) + criterion2(zcoef22, z22) + criterion2(zcoef31, z31) + criterion2(zcoef32, z32) 
        loss_r = criterion2(zoutput12, input1) + criterion2(zout12, input1) + criterion2(zoutput22, input2) + criterion2(zout22, input2) +  criterion2(zoutput32, input3) + criterion2(zout32, input3)

        loss_g4 = graph_loss(z12, coef) + graph_loss(z11, coef)
        loss_g5 = graph_loss(z22, coef) + graph_loss(z21, coef)
        loss_g6 = graph_loss(z32, coef) + graph_loss(z31, coef)
        loss_g = loss_g4+loss_g5+loss_g6

        loss_clustering = graph_loss(Q, coef)

        loss = loss_r + r1*loss_re + r1*loss_e + r2*loss_g + r1*loss_rc + 100*loss_clustering 
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    if epoch % 10 == 0:
        print("Epoch {}/{}".format(epoch, n_epochs2))
        print("Loss is:{:.4f}".format(loss.item()))
        print("Losse is:{:.4f}".format(loss_e.item()))
        print("Lossr is:{:.4f}".format(loss_r.item()))

        coef = torch.abs(coef)

        commonZ = coef.cpu().detach().numpy()
        alpha = max(0.4 - (10 - 1) / 10 * 0.1, 0.1)
        commonZ = thrC(commonZ, alpha)
        preds, _ = post_proC(commonZ, 10)

        label_preds = best_map(label_true, preds)

        label_preds = torch.LongTensor(label_preds)
        label_preds = label_preds.to(device)
        label_preds = label_preds - torch.ones_like(label_preds)
        Q = F.one_hot(label_preds, num_classes=10)
        Q = Q.to(dtype=torch.float)
        Q = Q.to(device)

        acc = metrics.acc(label_true, preds)
        nmi = metrics.nmi(label_true, preds)
        fscore = metrics.f_score(label_true, preds)
        ari = metrics.ari(label_true, preds)
        precision = metrics.b3_precision_score(label_true, preds)
        recall = metrics.b3_recall_score(label_true, preds)

        ACC_FM[0, int(epoch / 10)] = acc
        NMI_FM[0, int(epoch / 10)] = nmi
        FSCORE_FM[0, int(epoch / 10)] = fscore
        ARI_FM[0, int(epoch / 10)] = ari
        PRECISION_FM[0, int(epoch / 10)] = precision
        RECALL_FM[0, int(epoch / 10)] = recall

        print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f, ari: %.4f, precision: %.4f, recall: %.4f,  fscore: %.4f   <==|'
            % (acc, nmi, ari, precision, recall, fscore))
sio.savemat('NMI_FM' + '.mat', {'NMI_FM': NMI_FM})
sio.savemat('ACC_FM' +'.mat', {'ACC_FM': ACC_FM})
sio.savemat('FSCORE_FM' +'.mat', {'FSCORE_FM': FSCORE_FM})
sio.savemat('ARI_FM' +'.mat', {'ARI_FM': ARI_FM})
sio.savemat('PRECISION_FM' +'.mat', {'PRECISION_FM': PRECISION_FM})
sio.savemat('RECALL_FM' +'.mat', {'RECALL_FM': RECALL_FM})


