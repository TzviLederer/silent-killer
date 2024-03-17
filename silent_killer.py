import logging
from argparse import ArgumentParser
from random import seed

import numpy as np
import pandas as pd
import torch

import wandb
from utils import utils
from utils.crafter import PoisonCrafter
from utils.datasets import TriggeredDataset
from utils.trainer import evaluate_poisoning
from utils.utils import init_wandb, load_data, log_to_file, DEBUG

logging.basicConfig(level=logging.INFO)
seed(42)
torch.manual_seed(42)
np.random.seed(42)


def get_args():
    arg_parser = ArgumentParser()
    # logging settings
    arg_parser.add_argument('--wandb', default='False')
    arg_parser.add_argument('--entity', default=None)
    arg_parser.add_argument('--project', default='silent-killer')
    arg_parser.add_argument('--name', default='base')
    arg_parser.add_argument('--train_print_freq', default=5, type=int)

    # run settings
    arg_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # attacker settings
    arg_parser.add_argument('-s', '--source_label', type=int, default=0, help='source class')
    arg_parser.add_argument('-t', '--target_label', type=int, default=3, help='target class')

    arg_parser.add_argument('-p', '--model_path', type=str, default=None,
                            help='If none, the initial model will be train from scratch, '
                                 'else it will be loaded from this path')
    arg_parser.add_argument('-l', '--log_path', type=str, default=None,
                            help='if not none, the main results will be writen to this file')

    # poison crafting
    arg_parser.add_argument('--n_samples', type=int, default=500, help='number of samples for poison')
    arg_parser.add_argument('--crafting_repetitions', type=int, default=250, help='epochs for poison crafting')
    arg_parser.add_argument('--alpha_poison', type=float, default=1 / 255,
                            help='step size of poison in gradient alignment')
    arg_parser.add_argument('--alpha_trigger', type=float, default=0 / 255,
                            help='step size of trigger in gradient alignment')
    arg_parser.add_argument('--poison_selection', type=str, default='gradient', choices=['random', 'gradient'])
    arg_parser.add_argument('--eps', type=float, default=16/255,
                            help='if the trigger is additive, choose small eps. e.g. 16/255, else set 1.')
    arg_parser.add_argument('--perturbations_norm', default='n_inf', choices=['l0', 'l2', 'l_inf'])

    arg_parser.add_argument('--retraining_factor', default=1, type=int,
                            help='the surrogate model will be retrained (retraining_factor - 1) during crafting')
    arg_parser.add_argument('--retraining_batch_size', default=128, type=int,
                            help='if retrain model during poison crafting')
    arg_parser.add_argument('--retraining_epochs', default=40, type=int, help='if retrain model during poison crafting')
    arg_parser.add_argument('--retraining_loss', default='cross_entropy',
                            help='if retrain model during poison crafting')

    # trigger optimization
    arg_parser.add_argument('--trigger_type', type=str, default='additive', choices=['additive', 'adaptive_patch'])
    arg_parser.add_argument('--trigger_opt', type=str, default='True')
    arg_parser.add_argument('--patch_path', type=str, default='./trigger.png',
                            help='if using patch as trigger, and loading the patch from file')
    arg_parser.add_argument('--trigger_loc', type=str, default='rand',
                            help='can be tuple of (y, x) or `rand`. used if the trigger is a patch')
    arg_parser.add_argument('--trigger_init_method', type=str, default='randn', choices=['from_file', 'randn'],
                            help='initialization of patch trigger, can be `from_file` or `randn` (random init)')
    arg_parser.add_argument('--trigger_batch_size', type=int, default=8_192)
    arg_parser.add_argument('--patch_size', type=int, default=8, help='patch size if the trigger is a patch')
    arg_parser.add_argument('--trigger_opt_epochs', type=int, default=500,
                            help='epochs num in pre-training trigger optimization, 500')
    arg_parser.add_argument('--trigger_opt_eps', type=float, default=16 / 255)
    arg_parser.add_argument('--trigger_opt_lr', type=float, default=10 / 255)
    arg_parser.add_argument('--trigger_opt_gamma', type=float, default=1, help='scheduler gamma')

    # victim training
    arg_parser.add_argument('--victim_milestones', type=int, default=(14, 24, 35), nargs='+',
                            help='scheduler milestones')
    arg_parser.add_argument('--victim_gamma', type=float, default=0.1, help='scheduler milestones')
    arg_parser.add_argument('--victim_lr', type=float, default=0.1)
    arg_parser.add_argument('--victim_momentum', type=float, default=0.9)
    arg_parser.add_argument('--victim_weight_decay', type=float, default=5e-4)
    arg_parser.add_argument('--victim_loss', default='cross_entropy')
    arg_parser.add_argument('--victim_optimizer', default='nesterov')
    arg_parser.add_argument('--victim_batch_size', default=128, type=int)
    arg_parser.add_argument('--victim_epochs', default=80, type=int)
    arg_parser.add_argument('--validation_frequency', default=5, type=int)
    arg_parser.add_argument('--victim_augmentations', default='True', type=str,
                            help='convet to bool, `true` or `false`')

    # evaluation
    arg_parser.add_argument('--craft_model', type=str, default='resnet18',
                            choices=['resnet18', 'vgg11', 'mobilenet_v2'])
    arg_parser.add_argument('--eval_model', type=str, default='resnet18', choices=['resnet18', 'vgg11', 'mobilenet_v2'])
    arg_parser.add_argument('--val_repetitions', type=int, default=1, help='number of repetitions of evaluation')

    # defences
    arg_parser.add_argument('--apply_defences', default='False')
    arg_parser.add_argument('--activation_clustering', default='False')
    arg_parser.add_argument('--mixup', default='False')
    arg_parser.add_argument('--dpsgd_clip', type=float, default=None,
                            help='dp-sgd defence - clip gradients during training')
    arg_parser.add_argument('--dpsgd_noise', type=float, default=0,
                            help='dp-sgd defence - add noise to gradients during training')

    args = arg_parser.parse_args()
    str2bool(args)
    return args


def str2bool(args):
    for k, val in args.__dict__.items():
        if isinstance(val, str) and val.lower() in ['true', 'false']:
            args.__dict__[k] = args.__dict__[k].lower() == 'true'


def get_defences(args):
    defences = {'apply_defences': args.apply_defences,
                'dpsgd_clip': args.dpsgd_clip, 'dpsgd_noise': args.dpsgd_noise,
                'activation_clustering': args.activation_clustering, 'mixup': args.mixup}
    return defences


def main():
    args = get_args()
    defences = get_defences(args)
    device = utils.get_device(args.device)

    if args.wandb:
        init_wandb(entity=args.entity, project=args.project, name=args.name, args=args)
    dataset_train, dataset_test, normalizer = load_data()

    print('initiating crafter')
    poison_crafter = PoisonCrafter(model_initializer=args.craft_model, model_path=args.model_path,
                                   clean_dataset=dataset_train, test_dataset=dataset_test,
                                   normalizer=normalizer, augmentations=args.victim_augmentations,
                                   source_label=args.source_label, target_label=args.target_label,
                                   n_samples=args.n_samples, victim_loss=args.victim_loss,
                                   victim_optimizer=args.victim_optimizer, victim_batch_size=args.victim_batch_size,
                                   victim_lr=args.victim_lr, victim_momentum=args.victim_momentum,
                                   victim_weight_decay=args.victim_weight_decay,
                                   victim_milestones=args.victim_milestones, poison_selection=args.poison_selection,
                                   victim_gamma=args.victim_gamma, crafting_repetitions=args.crafting_repetitions,
                                   alpha_poison=args.alpha_poison, alpha_trigger=args.alpha_trigger,
                                   trigger_batch_size=args.trigger_batch_size, patch_size=args.patch_size,
                                   trigger_init_method=args.trigger_init_method, log_wandb=args.wandb,
                                   trigger_type=args.trigger_type, device=device,
                                   eps_p=args.eps, eps_t=args.trigger_opt_eps,
                                   patch_path=args.patch_path, trigger_loc=args.trigger_loc,
                                   train_print_freq=args.train_print_freq, norm=args.perturbations_norm,
                                   retraining_factor=args.retraining_factor, retraining_loss=args.retraining_loss,
                                   retraining_batch_size=args.retraining_batch_size,
                                   retraining_epochs=args.retraining_epochs)
    if args.trigger_opt:
        print('optimizing trigger')
        poison_crafter.optimize_trigger(epochs=args.trigger_opt_epochs, eps=args.trigger_opt_eps,
                                        lr=args.trigger_opt_lr, gamma=args.trigger_opt_gamma)

    print('craft poison')
    poisoned_dataset = poison_crafter.craft()
    test_trigger_set = TriggeredDataset(clean_dataset=dataset_test,
                                        source_label=args.source_label, target_label=args.target_label,
                                        trigger_fn=poison_crafter.trigger_set.trigger_fn)
    test_trigger_set.label_to_return = 'target'

    print('evaluate results')
    n = args.val_repetitions if not DEBUG else 1
    results = pd.DataFrame(
        [evaluate_poisoning(poisoned_dataset, dataset_test, test_trigger_set, normalizer,
                            milestones=args.victim_milestones,
                            gamma=args.victim_gamma, model=args.eval_model, loss=args.victim_loss,
                            optimizer=args.victim_optimizer, momentum=args.victim_momentum,
                            batch_size=args.victim_batch_size, weight_decay=args.victim_weight_decay,
                            epochs=args.victim_epochs, lr=args.victim_lr, device=device,
                            validation_frequency=args.validation_frequency, defences=defences, log_wandb=args.wandb)
         for _ in range(n)])
    mean_results = {f'mean {k}': v for k, v in dict(results.mean()).items()}
    if args.wandb:
        wandb.log({'results': wandb.Table(dataframe=results), **mean_results})

    if args.log_path is not None:
        log_to_file(args, mean_results)


if __name__ == '__main__':
    main()

