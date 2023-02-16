import torch

label_tensor = torch.randn(3,224,224).unsqueeze(1)
# label_tensor[1] = 1.
# label_tensor = torch.nn.functional.one_hot(label_tensor, num_classes=142)
print(label_tensor.shape)
# label_tensor.scatter_(1)
# print(label_tensor)