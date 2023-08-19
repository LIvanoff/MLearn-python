import torch


def RMSE(pred, y):
    return torch.sqrt(torch.mean(torch.pow((pred - y), 2)))


def r2_score(pred, y):
    return 1 - torch.mean(torch.pow((pred - y), 2)) / torch.mean(
        torch.pow((y - torch.mean(y)), 2))


def r1_score(pred, y):
    return torch.sqrt(
        1 - torch.mean(torch.pow((pred - y), 2)) / torch.mean(torch.pow((y - torch.mean(y)), 2)))
