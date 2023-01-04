import os
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEBUG = False


def get_device(device):
    if device == 'cpu':
        print('using CPU')
        return torch.device('cpu')
    elif device == 'cuda':
        if torch.cuda.is_available():
            print('using CUDA')
            return torch.device('cuda')
        else:
            print('cannot find CUDA, using CPU')
            return torch.device('cpu')
    else:
        raise NotImplementedError('The device has to be either `cpu` or `cuda`')


def load_data():
    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    normalizer = get_normalizer(dataset_train)
    return dataset_train, dataset_test, normalizer


def init_wandb(entity, project, name, args=None):
    config = {}
    if args is not None:
        config.update(args.__dict__)

    config = {k: str(val) for k, val in config.items()}
    try:
        slurm_id = os.environ['SLURM_JOB_ID']
        config.update({'job_id': int(slurm_id)})
        print(f'job id is: {slurm_id}')
    except KeyError:
        pass
    wandb.init(entity=entity, project=project, name=name, config=config)


def get_normalizer(dataset, n=256):
    mean = next(iter(DataLoader(dataset, batch_size=n)))[0].mean(dim=(0, 2, 3))
    std = next(iter(DataLoader(dataset, batch_size=n)))[0].std(dim=(0, 2, 3))
    print(f'Normalization computed mean: {mean}, std: {std}')
    return transforms.Normalize(mean, std)


def init_optimizer(optimizer, parameters, lr, momentum=None, decay=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    if optimizer == 'adam':
        optimizer = optim.Adam(lr=lr, params=parameters)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(lr=lr, params=parameters)
    elif optimizer == 'nesterov':
        optimizer = optim.SGD(lr=lr, params=parameters, nesterov=True, momentum=momentum, weight_decay=decay)
    else:
        raise NotImplementedError
    return optimizer


def init_loss(loss):
    if loss == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss == 'mse':
        loss_fn = nn.MSELoss()
    elif loss == 'l1':
        loss_fn = nn.L1Loss()
    else:
        raise NotImplementedError
    return loss_fn


def resnet18(pretrained=False, out_features=10):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18()
    return nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten(), nn.Linear(512, out_features))


def vgg11(pretrained=False, out_features=10):
    if pretrained:
        model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    else:
        model = models.vgg11()
    head = list(model.children())[0]
    flatten = nn.Flatten()
    linear1 = nn.Linear(in_features=512, out_features=512)
    relu1 = nn.ReLU(inplace=True)
    do1 = nn.Dropout(p=0.5, inplace=False)
    linear2 = nn.Linear(512, 512)
    relu2 = nn.ReLU(inplace=True)
    do2 = nn.Dropout(p=0.5, inplace=False)
    output = nn.Linear(512, out_features)
    classifier = nn.Sequential(linear1, relu1, do1, linear2, relu2, do2, output)
    return nn.Sequential(head, flatten, classifier)


def gradient_matching(trigger_grads, poison_grads):
    """
    Compute the blind passenger loss term.
    """
    matching = 0
    poison_norm = 0
    source_norm = 0

    for poison_grad, trigger_grad in zip(poison_grads, trigger_grads):
        matching += (trigger_grad * poison_grad).sum()
        poison_norm += poison_grad.pow(2).sum()
        source_norm += trigger_grad.pow(2).sum()

    matching = matching / poison_norm.sqrt() / source_norm.sqrt()
    return 1 - matching


class VideoWriter:
    def __init__(self, out_path, codec='mp4v', fps=1, display=True):
        Path(out_path).parent.mkdir(exist_ok=True, parents=True)
        self.out_path = out_path
        self.video_writer = None
        self.fps = fps
        self.codec = codec
        self.display = display

    def write(self, im: torch.Tensor):
        if self.video_writer is None:
            shape = tuple(im.shape[1:3])
            codec = cv2.VideoWriter_fourcc(*self.codec)
            self.video_writer = cv2.VideoWriter(self.out_path, codec, self.fps, shape)

        im = im.clip(0, 1)
        im = im.permute(1, 2, 0)
        im = im.detach().numpy()
        im = (im * 255).astype(np.uint8)

        if self.display:
            try:
                plt.imshow(im)
                plt.pause(0.001)
            except AttributeError:
                print('\rCannot show figure', end='')

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        self.video_writer.write(im)

    def release(self):
        self.video_writer.release()


def log_to_file(args, mean_results):
    with open(args.log_path, 'a') as f:
        output = args.__dict__
        output.update(mean_results)
        f.write(f'{output}\n')


def mobilenet(pretrained=False, out_features=10):
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights)
    else:
        model = models.mobilenet_v2()
    head = list(model.children())[:-1]
    classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.2), nn.Linear(1280, out_features=out_features))
    model = nn.Sequential(*head, classifier)
    return model


class Mixup(torch.nn.Module):
    """This is data augmentation via mixup."""

    def __init__(self, nway=2, alpha=1.0):
        """Implement differentiable mixup, mixing nway-many examples with the given mixing factor alpha."""
        super().__init__()
        self.nway = nway
        self.mixing_alpha = alpha

    def forward(self, x, y, epoch=None):
        if self.mixing_alpha > 0:
            lmb = np.random.dirichlet([self.mixing_alpha] * self.nway, size=1).tolist()[0]
            batch_size = x.shape[0]
            indices = [torch.randperm(batch_size, device=x.device) for _ in range(self.nway)]
            mixed_x = sum(l * x[index, :] for l, index in zip(lmb, indices))
            y_s = [y[index] for index in indices]
        else:
            mixed_x = x
            y_s = y
            lmb = 1

        return mixed_x, y_s, lmb

    def corrected_loss(self, outputs, extra_labels, lmb=(1.0,), loss_fn=torch.nn.CrossEntropyLoss()):
        """Compute the corrected loss under consideration of the mixing."""
        predictions = torch.argmax(outputs.data, dim=1)
        correct_preds = sum([w * predictions.eq(l.data).cpu().numpy().mean() for w, l in zip(lmb, extra_labels)])
        loss = sum(weight * loss_fn(outputs, label) for weight, label in zip(lmb, extra_labels))
        return loss, correct_preds


def get_model(model):
    if model == 'resnet18':
        model = resnet18()
    elif model == 'vgg11':
        model = vgg11()
    elif model == 'mobilenet_v2':
        model = mobilenet()
    return model
