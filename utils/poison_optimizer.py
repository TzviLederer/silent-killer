import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from utils import datasets as ds_utils
from utils.utils import resnet18, load_data


def optimize_poison(model, dataset, normalizer, trigger, source_label, target_label, patch_size, trigger_loc,
                    device='cuda', batch_size=256, lr=0.01, epochs=100, scheduler_step=200, scheduler_gamma=0.1,
                    verbose=1, **kwargs):
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # prepare dataset
    source_indexes = get_label_indexes(dataset=dataset, label=source_label)
    target = torch.ones(batch_size, dtype=torch.long) * target_label
    target = target.to(device)

    injector = ds_utils.TriggerAdaptivePatch('from_tensor', trigger=trigger, patch_size=patch_size, trigger_loc=trigger_loc)

    optimizer = torch.optim.Adam(params=[injector.trigger], lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    history = []
    for i in range(epochs):
        optimizer.zero_grad()

        triggered = sample_batch(batch_size, dataset, injector, source_indexes)
        loss, pred = optimization_step(model, triggered, normalizer, target, optimizer, scheduler, loss_fn, device)

        mean_sm = pred.softmax(dim=1).mean(dim=0).detach().cpu().numpy()
        loss_i = loss.detach().cpu().tolist()
        history.append((loss_i, mean_sm[source_label], mean_sm[target_label]))
        if verbose > 0:
            print(f'\r[{i + 1}] loss: {loss_i:.4f} | SM source {mean_sm[source_label]:.4f} | '
                  f'SM target {mean_sm[target_label]:.4f}', end='')
    return injector.trigger.detach()


def optimize_poison_additive(model, dataset, normalizer, device, source_label, target_label, eps, lr=1 / 255,
                             batch_size=256, epochs=500, **kwargs):
    # Create initial poison
    im, _ = dataset[0]
    noise = torch.zeros_like(im, requires_grad=True)

    # find source indexes and make the target label
    source_indexes = [i for i, (_, t) in enumerate(dataset) if t == source_label]
    target = torch.ones(batch_size, dtype=torch.long) * target_label

    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    for i in range(epochs):
        # make poisoned input samples
        inds = np.random.choice(source_indexes, size=batch_size, replace=False)
        triggered = [dataset[ind][0] + noise for ind in inds]
        triggered = torch.stack(triggered)

        # training step
        pred = model(normalizer(triggered).to(device))
        loss = loss_fn(pred, target.to(device))
        grads = torch.autograd.grad(loss, noise)
        noise = noise - grads[0].sign() * lr
        noise = noise.detach().clip(-eps, eps)
        noise.requires_grad_()

        # history
        mean_sm = pred.softmax(dim=1).mean(dim=0).detach().cpu().numpy()
        loss_i = loss.detach().cpu().tolist()
        print(
            f'\r[{i + 1}] loss: {loss_i:.4f} | SM source {mean_sm[source_label]:.4f} | SM target {mean_sm[target_label]:.4f} | {noise.abs().max():.5f}',
            end='')
    return noise.detach()


def optimization_step(model, batch, normalizer, target, optimizer, scheduler, loss_fn, device):
    pred = model(normalizer(batch).to(device))
    loss = loss_fn(pred, target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss, pred


def sample_batch(batch_size, dataset, injector, source_indexes):
    inds = np.random.choice(source_indexes, size=batch_size, replace=False)
    triggered = [injector.inject(dataset[ind][0]) for ind in inds]
    triggered = torch.stack(triggered)
    return triggered


def get_label_indexes(dataset, label):
    labels = np.array(list(map(lambda x: x[1], dataset)))
    indexes = np.where(labels == label)[0]
    return indexes
