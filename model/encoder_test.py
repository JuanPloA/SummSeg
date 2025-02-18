import torch
import torch.nn as nn

class SumCapEncoderLayer(nn.Module):
    # Implementación de la capa SumCapEncoderLayer aquí
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SumCapEncoderLayer, self).__init__()
        # Inicializa tus capas aquí

    def forward(self, x):
        # Implementa la lógica para pasar por esta capa
        return x, None, x.mean(dim=1)  # Retorna la salida y el resumen

class SumCapEncoder(nn.Module):
    def __init__(self, d_inp=1024, n_layers=4, n_head=1, d_k=64, d_v=64,
                 d_model=256, d_inner=512, dropout=0.1, use_drop_out=False, use_layer_norm=False):
        super(SumCapEncoder, self).__init__()
        self.n_layers = n_layers
        self.use_drop_out = use_drop_out
        self.use_layer_norm = use_layer_norm
        
        self.proj = nn.Linear(d_inp, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            SumCapEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.linear_1 = nn.Linear(d_model, d_model)  # Cambiar para usar d_model
        self.linear_2 = nn.Linear(d_model, 1)  # Cambiar para usar d_model

        self.drop = nn.Dropout(p=0.5)
        self.norm_linear = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)  # Cambiar para usar d_model
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src_seq):
        # Asegúrate de que src_seq tenga forma (300, 1024)
        enc_output = self.proj(src_seq)  # (300, d_model)
        enc_output = self.layer_norm(enc_output)  # (300, d_model)

        for enc_layer in self.layer_stack:
            enc_output, _, s = enc_layer(enc_output)  # `s` se espera que tenga forma (300, d_model)
        
        # 2 layer NN for regressor
        s = self.linear_1(s)  # Cambiar d_inp a d_model
        s = self.relu(s)  # Aplicar ReLU
        
        if self.use_drop_out:
            s = self.drop(s)
        if self.use_layer_norm:
            s = self.norm_linear(s)

        s = self.linear_2(s)  # Produciendo la salida final
        s = self.sigmoid(s)
        s = s.squeeze(-1)  # Eliminar dimensión adicional

        return enc_output, s  # `enc_output` (300, d_model) y `s` (300,)

# Ejemplo de uso
model = SumCapEncoder()
inp = torch.rand(300, 1024)  # Forma correcta de entrada
enc_output, s = model(inp)
print(enc_output.shape)  # Debería imprimir (300, d_model)
print(s.shape)  # Debería imprimir (300,)
