import torch
import torchvision

# model = torchvision.models.resnet50(pretrained=True)
model=torchvision.models.resnet.ResNet( torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
model.load_state_dict(torch.load("resnet50-19c8e357.pth"))
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet50.pt")
