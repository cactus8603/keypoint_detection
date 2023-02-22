from torch import nn

class TransformerAutoEncoder(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x 

class VAE(nn.Module):
    def __init__(self,):
        super().__init__()

