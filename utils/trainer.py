from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.cuda
import wandb
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop
from tqdm import tqdm

from utils.utils import init_optimizer, init_loss, load_data, mobilenet, Mixup, get_model, DEBUG


class Trainer:
    def __init__(self, model, device, normalizer, loss, optimizer, momentum, batch_size, weight_decay, epochs, lr,
                 milestones, gamma, validation_frequency, augmentations=True, defences=None):
        """
        :param model:       a pytorch model
        :param device:      torch.device
        :param normalizer:  torchvision transform
        """
        # training params
        self.loss = loss
        self.optimizer_name = optimizer
        self.momentum = momentum
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr = lr
        self.milestones = milestones
        self.gamma = gamma
        self.validation_frequency = validation_frequency
        if augmentations:
            self.augmentations = Compose([RandomHorizontalFlip(p=0.5), RandomCrop(32, padding=4)])
        else:
            self.augmentations = None

        self.device = device
        self.model = model.to(self.device)
        self.best_model = model
        self.best_epoch = None

        self.loss_fn = init_loss(loss)
        self.dataset_train = None
        self.dataset_test = None
        self.normalizer = normalizer
        self.optimizer = None

        self.defences = defences
        if defences['mixup']:
            self.mixup = Mixup()

    def fit(self, dataset, dataset_test: dict = (), early_stop_dataset=None):
        self.dataset_train = dataset
        self.dataset_test = dataset_test
        best_loss = np.inf
        best_acc = 0

        self.optimizer = init_optimizer(self.optimizer_name, self.model.parameters(), self.lr,
                                        decay=self.weight_decay, momentum=self.momentum)
        scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=self.gamma)

        for epoch in range(self.epochs):
            self.fit_epoch(epoch)
            scheduler.step()

            test_set_exists = self.dataset_test is not None
            eval_step = (epoch + 1) % self.validation_frequency == 0
            final_step = epoch == self.epochs - 1
            if test_set_exists and (eval_step or final_step):
                for i, (name, val_set) in enumerate(self.dataset_test.items()):
                    # eval dataset
                    acc, loss = self.eval(val_set, name=name)

                    # saving the best model
                    if early_stop_dataset == i and loss < best_loss:
                        self.best_model = deepcopy(self.model)
                        self.best_epoch = epoch
                        best_acc = acc
                        best_loss = loss
            print('')

        print('Training complete.', end=' ')
        if self.best_epoch is None:
            print('The final model is the last one.')
        else:
            print(f'The final model was from epoch #{self.best_epoch + 1}, '
                  f'with best loss of {best_loss:.4f} and best accuracy of {best_acc:.4f}')
        self.model = self.best_model

    def eval(self, validation_set, name='test'):
        self.model.eval()
        epoch_loss_list = []
        epoch_pred = []
        for x_test, y_test in DataLoader(validation_set, batch_size=self.batch_size):
            x_test, y_test = x_test.to(self.device), y_test.to(self.device)
            loss, pred = self.eval_step(x_test, y_test)
            epoch_loss_list.append(loss.cpu().detach().tolist())
            epoch_pred.append(pred)
        predictions = np.concatenate(list(map(lambda x: np.array(x[0]), epoch_pred)))
        ys = np.concatenate(list(map(lambda x: np.array(x[1]), epoch_pred)))
        acc = accuracy_score(ys, predictions)
        print(f' | {name} loss: {np.array(epoch_loss_list).mean():.4f} | {name} acc: {acc:.4f}', end='')
        return acc, np.array(epoch_loss_list).mean()

    def fit_epoch(self, epoch):
        self.model.train()
        epoch_loss_list = []
        epoch_pred = []
        for x_train, y_train in DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True):
            self.fit_batch(epoch, epoch_loss_list, epoch_pred, x_train, y_train)
            if DEBUG:
                break
        predictions = np.concatenate(list(map(lambda x: np.array(x[0]), epoch_pred)))
        ys = np.concatenate(list(map(lambda x: np.array(x[1]), epoch_pred)))
        acc = accuracy_score(ys, predictions)
        print(f' | acc: {acc:.4f}', end='')

    def fit_batch(self, epoch, epoch_loss_list, epoch_pred, x_train, y_train):
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)

        if self.augmentations is not None:
            x_train = self.augmentations(x_train)

        if self.defences['apply_defences'] and self.defences['mixup']:
            x_train, y_train, self.lmb = self.mixup(x_train, y_train)

        loss, pred = self.training_step(x_train, y_train)
        epoch_loss_list.append(loss.cpu().detach().tolist())
        epoch_pred.append(pred)
        print(f'\repoch [{epoch + 1}/{self.epochs}], loss: {np.array(epoch_loss_list).mean():.4f}', end='')

    def training_step(self, x_train, y_train):
        self.optimizer.zero_grad()
        pred = self.model(self.normalizer(x_train))

        if self.defences['apply_defences'] and self.defences['mixup']:
            loss, acc = self.mixup.corrected_loss(outputs=pred, extra_labels=y_train, lmb=self.lmb)
            y_train = y_train[0]
        else:
            loss = self.loss_fn(pred, y_train)
        loss.backward()

        # apply defences
        if self.defences['apply_defences']:
            self.apply_dpsgd()

        self.optimizer.step()
        return loss, (pred.argmax(dim=1).cpu().tolist(), y_train.cpu().tolist())

    def apply_dpsgd(self):
        apply_clip = 'dpsgd_clip' in self.defences.keys() and self.defences['dpsgd_clip'] is not None
        if apply_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.defences['dpsgd_clip'])
        if 'pdsgd_noise' in self.defences.keys():
            loc = torch.as_tensor(0., device=self.device)
            clip_factor = self.defences['dpsgd_clip'] if apply_clip else 1.0
            scale = torch.as_tensor(clip_factor * self.defences['pdsgd_noise'], device=self.device)
            generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            for param in self.model.parameters():
                param.grad += generator.sample(param.shape)

    def eval_step(self, x_test, y_test):
        pred = self.model(self.normalizer(x_test))
        loss = self.loss_fn(pred, y_test)
        return loss, (pred.argmax(dim=1).cpu().tolist(), y_test.cpu().tolist())


def evaluate_poisoning(dataset_poisoned, dataset_test, dataset_trigger, normalizer, loss, optimizer, momentum,
                       batch_size, weight_decay, epochs, lr, milestones, gamma, validation_frequency, log_wandb,
                       device, model='resnet18', defences=None):
    if defences['apply_defences'] and defences['activation_clustering']:
        activation_clustering(batch_size, dataset_poisoned, dataset_test, dataset_trigger, defences, epochs,
                              gamma, loss, lr, milestones, model, momentum, normalizer, optimizer,
                              validation_frequency, weight_decay, device=device)
    model = get_model(model)
    trainer = Trainer(model=model, device=device, normalizer=normalizer, loss=loss, optimizer=optimizer,
                      momentum=momentum, batch_size=batch_size, weight_decay=weight_decay, epochs=epochs, lr=lr,
                      milestones=milestones, gamma=gamma, validation_frequency=validation_frequency, defences=defences)
    trainer.fit(dataset_poisoned, {'clean_val': dataset_test, 'triggered_val': dataset_trigger}, early_stop_dataset=0)
    asr, trigger_set_loss = trainer.eval(dataset_trigger)
    clean_acc, clean_loss = trainer.eval(dataset_test)
    results = {'ASR': asr, 'Clean Acc': clean_acc,
               'clean loss': clean_loss, 'trigger set loss': trigger_set_loss}
    if log_wandb:
        wandb.log(results)
    return results


def main():
    from sklearn.model_selection import train_test_split

    model_output = 'models/mobilenet_cifar10.pt'

    class Params:
        loss = 'cross_entropy'
        optimizer = 'nesterov'
        momentum = 0.9
        batch_size = 128
        weight_decay = 4e-4
        epochs = 150
        lr = 0.1
        augmentations = True
        validation_frequency = 5
        milestones = [50, 100]
        gamma = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_train, dataset_test, normalizer = load_data()
    train_set, validation_set = train_test_split(dataset_train, random_state=42)
    model = mobilenet(pretrained=False)

    trainer = Trainer(model=model, normalizer=normalizer, device=device, loss=Params.loss, optimizer=Params.optimizer,
                      momentum=Params.momentum, batch_size=Params.batch_size, weight_decay=Params.weight_decay,
                      epochs=Params.epochs, lr=Params.lr, milestones=Params.milestones, gamma=Params.gamma,
                      validation_frequency=Params.validation_frequency, augmentations=Params.augmentations)
    trainer.fit(train_set, {'validation': validation_set}, 0)
    trainer.eval(dataset_test, name='test set')

    if model_output is not None:
        torch.save(model.state_dict(), model_output)


def activation_clustering(batch_size, dataset_poisoned, dataset_test, dataset_trigger, defences, epochs, gamma, loss,
                          lr, milestones, model, momentum, normalizer, optimizer, validation_frequency, weight_decay,
                          device):
    print('train model for activation clustering')
    feature_extractor, model = train_feature_extractor(batch_size, dataset_poisoned, dataset_test, dataset_trigger,
                                                       defences, epochs, gamma, loss, lr, milestones, model,
                                                       momentum, normalizer, optimizer, validation_frequency,
                                                       weight_decay, device=device)
    print('extract features')
    class_indices, features = extract_features(dataset_poisoned, feature_extractor, device=device)
    print('find clean samples')
    clean_indices = filter_poison(class_indices, features)
    dataset_poisoned.enabled_indexes = clean_indices
    print(f'new size: {len(dataset_poisoned)}')


def train_feature_extractor(batch_size, dataset_poisoned, dataset_test, dataset_trigger, defences, epochs, gamma, loss,
                            lr, milestones, model, momentum, normalizer, optimizer, validation_frequency, weight_decay,
                            device):
    model = get_model(model)
    trainer = Trainer(model=model, device=device, normalizer=normalizer, loss=loss, optimizer=optimizer,
                      momentum=momentum, batch_size=batch_size, weight_decay=weight_decay, epochs=epochs, lr=lr,
                      milestones=milestones, gamma=gamma, validation_frequency=validation_frequency, defences=defences)
    trainer.fit(dataset_poisoned, {'clean_val': dataset_test, 'triggered_val': dataset_trigger}, early_stop_dataset=0)
    layer_cake = list(model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    return feature_extractor, model


def filter_poison(class_indices, features):
    clean_indices = []
    for i in class_indices.keys():
        if len(class_indices[i]) > 1:
            temp_feats = np.array([features[temp_idx].squeeze(0).cpu().numpy() for temp_idx in class_indices[i]])
            kmeans = KMeans(n_clusters=2).fit(temp_feats)
            clean_label = np.median(kmeans.labels_)
            clean = [class_indices[i][idx] for idx, is_clean in enumerate(kmeans.labels_ == clean_label) if is_clean]
            clean_indices = clean_indices + clean
    return clean_indices


def extract_features(dataset_poisoned, feature_extractor, device):
    features = []
    class_indices = defaultdict(list)
    with torch.no_grad():
        for i, (img, source) in tqdm(enumerate(dataset_poisoned), total=len(dataset_poisoned)):
            img = img.unsqueeze(0).to(device)
            features.append(feature_extractor(img))
            class_indices[source].append(i)
    return class_indices, features


if __name__ == '__main__':
    main()
