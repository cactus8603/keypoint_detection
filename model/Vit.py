from torch import nn

from .module import PatchEmbeddings, TransformerEncoder, ClassificationHead

class Vit(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.Vit = nn.Sequential(
            PatchEmbeddings(args_dict),
            TransformerEncoder(args_dict),
            ClassificationHead(args_dict),
        )

    def forward(self, x):
        return self.Vit(x)