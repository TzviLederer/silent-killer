from random import shuffle

import PIL
import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop

from utils import datasets as ds_utils
from utils.poison_optimizer import optimize_poison, optimize_poison_additive
from utils.utils import init_loss, init_optimizer, gradient_matching, resnet18, vgg11, mobilenet, DEBUG
from utils.aug import RandomTransform

def get_trigger_function(trigger_type, **kwargs):
    if trigger_type == 'additive':
        return ds_utils.TriggerAdditive(**kwargs)
    elif trigger_type == 'adaptive_patch':
        return ds_utils.TriggerAdaptivePatch(**kwargs)


class PoisonCrafter:
    def __init__(self, model_initializer, clean_dataset, test_dataset, source_label, target_label, n_samples,
                 victim_lr, victim_momentum, victim_weight_decay, victim_milestones, victim_gamma, victim_loss,
                 victim_batch_size, victim_optimizer, alpha_poison, alpha_trigger, crafting_repetitions,
                 poison_selection, trigger_batch_size, trigger_init_method, log_wandb, trigger_type, device, patch_path,
                 trigger_loc, eps_p, eps_t, retraining_factor, retraining_epochs, retraining_batch_size,
                 retraining_loss, normalizer: transforms = transforms.Compose([]), log_freq=10, augmentations=True,
                 model_path=None, patch_size=8, train_print_freq=5, norm='l_inf'):
        self.device = device
        self.log_wandb = log_wandb

        self.r = None
        self.log_freq = log_freq
        self.n_samples = n_samples
        self.alpha_poison = alpha_poison
        self.alpha_trigger = alpha_trigger
        self.crafting_repetitions = crafting_repetitions
        self.poison_selection = poison_selection  # 'random' or 'gradient'
        self.patch_size = (3, patch_size, patch_size)
        self.trigger_batch_size = trigger_batch_size
        self.eps_p = eps_p
        self.eps_t = eps_t
        self.norm = norm

        # victim training parameters
        self.victim_lr = victim_lr
        self.victim_momentum = victim_momentum
        self.victim_weight_decay = victim_weight_decay
        self.victim_milestones = victim_milestones
        self.victim_gamma = victim_gamma
        self.victim_batch_size = victim_batch_size
        self.victim_optimizer = victim_optimizer
        self.train_print_freq = train_print_freq

        # retraining parameters
        self.retraining_factor = retraining_factor
        self.retraining_epochs = retraining_epochs
        self.retraining_batch_size = retraining_batch_size
        self.retraining_loss = retraining_loss

        # training settings
        if model_initializer == 'resnet18':
            self.model_initializer = resnet18
        elif model_initializer == 'vgg11':
            self.model_initializer = vgg11
        elif model_initializer == 'mobilenet_v2':
            self.model_initializer = mobilenet
        else:
            raise NotImplementedError
        self.loss_fn = init_loss(self.retraining_loss)
        self.victim_loss = init_loss(victim_loss)

        if augmentations:
            # self.augmentations = Compose([RandomHorizontalFlip(p=0.5), RandomCrop(32, padding=4)])
            self.augmentations = RandomTransform(source_size=32, target_size=32, shift=8, fliplr=True)
        else:
            self.augmentations = transforms.Compose([])

        # data
        self.clean_dataset = clean_dataset
        self.test_dataset = test_dataset
        self.normalizer = normalizer

        # Create trigger set
        self.trigger_type = trigger_type
        self.source_class = source_label
        self.target_label = target_label
        self.patch_path = patch_path
        self.trigger_loc = trigger_loc

        self.trigger_injector = get_trigger_function(trigger_type, size=clean_dataset[0][0].shape, sigma=0.01,
                                                     patch_size=self.patch_size, method=trigger_init_method,
                                                     patch_path=patch_path, trigger_loc=trigger_loc)
        self.trigger_set = ds_utils.TriggeredDataset(clean_dataset=clean_dataset,
                                                     source_label=source_label, target_label=target_label,
                                                     trigger_fn=self.trigger_injector.inject)
        # create model
        self.model, self.optimizer, self.scheduler = self._init_model(lr=victim_lr, momentum=victim_momentum,
                                                                      weight_decay=victim_weight_decay,
                                                                      gamma=self.victim_gamma,
                                                                      milestones=self.victim_milestones)
        if model_path is not None:
            print('load model from file')
            self.model.load_state_dict(torch.load(model_path))
        else:
            print('train initial model from scratch')
            self._train(self.clean_dataset)

        # Create poison set
        indexes = self._choose_indexes()
        self.poisoned_dataset = ds_utils.PoisonedDataset(clean_dataset=clean_dataset, eps=self.eps_p,
                                                         norm=self.norm, indexes=indexes)

    def optimize_trigger(self, epochs, eps, lr, gamma):
        trigger = self.trigger_injector.trigger
        if self.trigger_type == 'adaptive_patch':
            opt_fn = optimize_poison
        elif self.trigger_type == 'additive':
            opt_fn = optimize_poison_additive
        else:
            raise NotImplementedError
        optimized_trigger = opt_fn(model=self.model, dataset=self.clean_dataset, normalizer=self.normalizer,
                                   device=self.device, trigger=trigger, source_label=self.source_class,
                                   trigger_loc=self.trigger_loc, epochs=epochs, eps=eps, lr=lr,
                                   target_label=self.target_label, patch_size=self.patch_size,
                                   trigger_type=self.trigger_type, scheduler_gamma=gamma)
        self.trigger_injector.update_trigger(optimized_trigger)

    def _choose_indexes(self):
        method = self.poison_selection
        if method == 'random':
            indexes = self._random_poison_selection()
        elif method == 'gradient':
            indexes = self._gradient_based_poison_selection(out_of_class=False, model=self.model)
        elif method == 'out_of_class_random':
            indexes = self._random_ooc_poison_selection()
        elif method == 'out_of_class_gradient':
            indexes = self._gradient_based_poison_selection(out_of_class=True, model=self.model)
        else:
            raise NotImplementedError
        print(f'Poisoning set, {len(indexes)}/{len(self.clean_dataset)} samples were randomly chose to poison, '
              f'{len(indexes) / len(self.clean_dataset) * 100:.2f}% of the dataset.')
        return indexes

    def _random_poison_selection(self):
        print('Random poison selection')
        indexes = [i for i, x in enumerate(self.clean_dataset) if x[1] == self.target_label]
        shuffle(indexes)
        indexes = indexes[:self.n_samples]
        return indexes

    def _random_ooc_poison_selection(self):
        """
        Choose poison samples of other classes than the target class
        """
        print('Random out-of-class poison selection')
        indexes = [i for i, x in enumerate(self.clean_dataset) if x[1] != self.target_label]
        shuffle(indexes)
        indexes = indexes[:self.n_samples]
        return indexes

    def _gradient_based_poison_selection(self, model, out_of_class=False):
        print('Gradient based poison selection')
        grad_dict = {}
        model.eval()
        parameters = [p for p in model.parameters() if p.requires_grad]
        for i, (x, y) in enumerate(DataLoader(self.clean_dataset, batch_size=1)):
            x, y = x.to(self.device), y.to(self.device)
            if i % 1000 == 0:
                print(f'\r[{i}/{len(self.clean_dataset)}] Choosing largest grad samples', end='')
            if (not out_of_class) and (y != self.target_label):
                continue
            elif out_of_class and (y == self.target_label):
                continue
            pred = model(self.normalizer(x))
            loss = self.victim_loss(pred, y)
            grads = torch.autograd.grad(loss, parameters)
            grads_norm = torch.stack(list(map(lambda p: p.norm(), grads))).sum()
            grad_dict[i] = grads_norm.tolist()
            if DEBUG:
                break
        grad_dict = dict(sorted(grad_dict.items(), key=lambda item: item[1], reverse=True))
        indexes = list(grad_dict.keys())[:self.n_samples]

        print('')
        return indexes

    def craft(self):
        r = self.crafting_repetitions  # R in the original paper
        t = self.retraining_factor  # T in the original paper
        for self.r in range(r):
            print(f'\nREPETITION {self.r + 1}:')
            if ((self.r + 1) % np.ceil(r / t) == 0) and ((self.r + 1) != r):
                self.model, self.optimizer, self.scheduler = self._init_model(lr=self.victim_lr,
                                                                              momentum=self.victim_momentum,
                                                                              weight_decay=self.victim_weight_decay,
                                                                              gamma=self.victim_gamma,
                                                                              milestones=self.victim_milestones)
                self._train(self.poisoned_dataset)

            self._optimize_poison()
            if DEBUG:
                break
        return self.poisoned_dataset

    def _optimize_poison(self):
        self.model.eval()

        batch_size = self.trigger_batch_size
        dataloader_train = DataLoader(self.trigger_set, batch_size=batch_size, shuffle=True)

        # training loop
        loss = 0
        for step, (x_batch, y_batch, y_target) in enumerate(dataloader_train):
            loss += self._compute_loss(x_batch, y_target)
            if DEBUG:
                break

        self._poison_optimization_step(loss)
        loss = loss.detach().cpu().tolist()
        print(f'\rPoison optimization | matching loss: {loss:.4f}')
        self.log_poisoning(loss=loss, log_wandb=self.log_wandb)

    def _poison_optimization_step(self, loss):
        grads = torch.autograd.grad(loss, (self.trigger_injector.trigger, self.poisoned_dataset.poison_subset.poison))

        trigger = self.trigger_injector.trigger.detach()
        trigger -= grads[0].sign() * self.alpha_trigger
        trigger = torch.clip(trigger, -self.eps_t, self.eps_t)
        trigger.requires_grad_()
        self.trigger_injector.trigger = trigger

        poison = self.poisoned_dataset.poison_subset.poison.detach()
        poison -= grads[1].sign() * self.alpha_poison
        poison = torch.clip(poison, -self.eps_p, self.eps_p)
        poison.requires_grad_()
        self.poisoned_dataset.poison_subset.poison = poison

    def _compute_loss(self, x_batch, y_target):
        poison_grads = self._compute_poison_gradient()
        trigger_grads = self._compute_trigger_gradient(x_batch, y_target)
        loss = gradient_matching(trigger_grads, poison_grads)
        return loss

    def log_poisoning(self, loss, log_wandb):
        im_p, _ = self.poisoned_dataset.poison_subset[0]
        im_c, _ = self.poisoned_dataset.poison_subset.clean_dataset[self.poisoned_dataset.poison_subset.indexes[0]]
        im_p, im_c = [im.cpu().detach().numpy().transpose((1, 2, 0)) for im in [im_p, im_c]]
        im_d = np.abs(im_p - im_c)

        im_t = [im for (im, _, _), _ in zip(self.trigger_set, range(3))]
        im_t = [im.cpu().detach().numpy().transpose((1, 2, 0)) for im in im_t]

        images = [PIL.Image.fromarray(np.uint8(image * 255)) for image in [im_p, im_c, im_d]]
        images_t = [PIL.Image.fromarray(np.uint8(image * 255)) for image in im_t]

        if log_wandb:
            wandb.log({"examples": [wandb.Image(image) for image in images],
                       'triggered_images': [wandb.Image(image) for image in images_t],
                       'max_diff': im_d.max(), 'diff_norm': np.linalg.norm(im_d), 'matching_loss': loss})

    def _compute_trigger_gradient(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model(self.normalizer(x))
        loss = self.victim_loss(pred, y)

        model_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.autograd.grad(loss, model_params, retain_graph=True, create_graph=True)

    def _compute_poison_gradient(self):
        loss = 0
        for data in DataLoader(self.poisoned_dataset.poison_subset, batch_size=self.victim_batch_size):
            x_poison, y_poison = list(map(lambda x: x.to(self.device), data))
            x_poison = self.augmentations(x_poison)
            pred = self.model(self.normalizer(x_poison))
            loss += self.victim_loss(pred, y_poison)
        loss /= len(self.poisoned_dataset)
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.autograd.grad(loss, model_params, retain_graph=True, create_graph=True)

    def _init_model(self, lr, momentum, weight_decay, gamma, milestones):
        model = self.model_initializer().to(self.device)
        optimizer = init_optimizer(self.victim_optimizer, model.parameters(), lr,
                                   momentum=momentum, decay=weight_decay)
        scheduler = MultiStepLR(optimizer, gamma=gamma, milestones=milestones)
        return model, optimizer, scheduler

    def _train(self, dataset):
        for epoch in range(self.retraining_epochs):
            loss_list = []
            dataloader = DataLoader(dataset, batch_size=self.retraining_batch_size, shuffle=True)
            for i, (x_batch, y_batch) in enumerate(dataloader):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.model.train()
                loss = self._train_step(x_batch, y_batch, augmentation=self.augmentations)
                loss_list.append(loss.cpu().detach().tolist())

                if (i + 1) % self.log_freq == 0:
                    print(f'\rEpoch {epoch + 1} | step {i + 1}/{len(dataset) // self.retraining_batch_size} | '
                          f'loss: {np.array(loss_list).mean():.5f}', end='')
                if DEBUG:
                    break
            if DEBUG:
                break

            if (epoch + 1) % self.train_print_freq == 0 or epoch == self.retraining_epochs - 1:
                self.print_metrics(dataset)

            self.scheduler.step()

    def print_metrics(self, dataset):
        loss_train, acc_train = self._compute_acc(dataset)
        loss_test, acc_test = self._compute_acc(self.test_dataset)
        self.trigger_set.label_to_return = 'target'
        loss_trigger, acc_trigger = self._compute_acc(self.trigger_set)
        self.trigger_set.label_to_return = 'both'

        print(f' | train: loss {loss_train:.4f}, acc {acc_train:.4f}', end='')
        print(f' | test: loss {loss_test:.4f}, acc {acc_test:.4f}', end='')
        print(f' | trigger: loss {loss_trigger:.4f}, ASR {acc_trigger:.4f}')

        if self.log_wandb:
            wandb.log({'r': self.r, 'loss_train': loss_train, 'acc_train': acc_train,
                       'loss_test': loss_test, 'acc_test': acc_test,
                       'loss_trigger': loss_trigger, 'acc_trigger': acc_trigger})

    def _compute_acc(self, dataset):
        self.model.eval()
        acc_list = []
        loss_list = []
        for x_batch, y_batch in DataLoader(dataset, batch_size=self.retraining_batch_size):
            loss, acc = self._eval_step(x_batch, y_batch)
            loss_list.append(loss)
            acc_list.extend(acc)

        loss = np.array(loss_list).mean()
        acc = np.array(acc_list).mean()

        return loss, acc

    def _train_step(self, x_batch, y_batch, augmentation=None):
        if augmentation is not None:
            x_batch = augmentation(x_batch)

        self.optimizer.zero_grad()
        pred = self.model(self.normalizer(x_batch))
        loss = self.loss_fn(pred, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def _eval_step(self, x_batch, y_batch):
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(self.normalizer(x_batch))
        loss = self.loss_fn(pred, y_batch)
        return loss.cpu().detach().tolist(), (pred.argmax(dim=-1) == y_batch).cpu().tolist()
