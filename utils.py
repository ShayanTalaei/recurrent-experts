from model import Expert

def init_weights(model: Expert, path: str) -> None:
    pre_trained_dict = torch.load(path, map_location=lambda storage, loc: storage)

    for layer in pre_trained_dict.keys():
        model.state_dict()[layer].copy_(pre_trained_dict[layer])

    for param in model.parameters():
        param.requires_grad = True
