import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import h5py

from .layers.encoder_layers import *

class SumCapEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_inp=1024, n_layers=4, n_head=4, d_k=64, d_v=64,
            d_model=1024, d_inner=2048, dropout=0.1, use_drop_out=True, use_layer_norm=True):


        #print("d_inp:", d_inp)
        #print("n_layers:", n_layers)
        #print("n_head:", n_head)
        #print("d_k:", d_k)
        #print("d_v:", d_v)
        #print("d_model:", d_model)
        #print("d_inner:", d_inner)
        #print("dropout:", dropout)
        #print("use_drop_out:", use_drop_out)
        #print("use_layer_norm:", use_layer_norm)
        super().__init__()
        # 4 Layers
        self.n_layers = n_layers
        #No usa dropout
        self.use_drop_out = use_drop_out
        #No usa layer norm
        self.use_layer_norm = use_layer_norm
        
        self.proj = None
        #Capa lineal de toda la vida: y = xW + b siendo x los datos con dimension d_imp, W los pesos de dimension d_model
        # d_inp = 1024 y d_model = 256
        self.proj = nn.Linear(d_inp, d_model) 
        #print("Pesos", self.proj.weight)
        #print("Bias", self.proj.bias)


        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            SumCapEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.linear_1 = nn.Linear(in_features=d_inp, out_features=d_inp)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.5)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, src_seq):
        # -- Forward
        
        #print("src_seq", src_seq)
        enc_output = self.proj(src_seq)
        #print("enc_output", enc_output)
        enc_output = self.layer_norm(enc_output)
        #print("Layers:", len(self.layer_stack))
        for i, enc_layer in enumerate(self.layer_stack):
            #print(i)
            enc_output, _, s = enc_layer(enc_output)
        #print("enc_output", enc_output)
        #s deben ser los saliency scores que pasa por el regresor para sacar los scores que me interesa
        # 2 layer NN for regressor
        s = self.linear_1(s)
        #print("s", s)
        s = self.relu(s)
        #print("s", s)
        if self.use_drop_out == True:
            s = self.drop(s)
            #print("s", s)
        if self.use_layer_norm == True:
            s = self.norm_linear(s)
            #print("s", s)

        s = self.linear_2(s)
        #print("s", s)
        s = self.sigmoid(s)
        #print("s", s)
        s = s.squeeze(-1)
        #print("S antes de devolver")
        #print(s)
            
        return enc_output, s

if __name__ == '__main__':
    seed_everything(42, workers=False)
    model = SumCapEncoder()
    
    model.eval()
    checkpoint_path = "epoch=054.ckpt"
    checkpoint = torch.load(checkpoint_path)
    encoder_ckpt = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith("bert")}
    encoder_ckpt = {k.replace("encoder.", ""): v for k, v in encoder_ckpt.items()}

    for keys in encoder_ckpt:
        print(keys)
    model.load_state_dict(encoder_ckpt)
    with h5py.File("../data/BIDS_swin_2s.h5", 'r+') as h5file:
        # 2. Acceder al grupo o nombre
        if 'v_IyUTgzwRmwg_60.0_210.0.mp4' in h5file:
            group = h5file['v_IyUTgzwRmwg_60.0_210.0.mp4']
            features = group['features']
            tensor = torch.tensor(features)
            tensor = tensor.unsqueeze(0)
            enc_output, s = model(tensor)
            print(s)
    
