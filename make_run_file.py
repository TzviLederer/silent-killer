from argparse import ArgumentParser

pairs = [(0, 1), (0, 3), (1, 6), (2, 0), (2, 1), (2, 7), (2, 9), (3, 2), (3, 7), (4, 1), (4, 2), (4, 6), (4, 9), (5, 2),
         (5, 3), (5, 6), (5, 9), (7, 2), (7, 9), (8, 2), (8, 4), (8, 6), (8, 9), (9, 7)]


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--wandb_entity', help='WandB entity for logging')
    parser.add_argument('--model_path', default=None, help='path to the downloaded weights')
    parser.add_argument('--model_arch', choices=['vgg11', 'resnet18', 'mobilenet'], help='architecture')
    parser.add_argument('--trigger_type', choices=['patch', 'additive'], help='architecture')
    parser.add_argument('--experiment', choices=['base', 'silent_killer'], help='architecture')
    return parser.parse_args()


def main():
    args = get_args()
    filename = f'run-{args.trigger_type}-{args.experiment}-{args.model_arch}.sh'
    with open(filename, 'w') as f:
        for s, t in pairs:
            f.write(get_command(s, t, args))


def get_command(s, t, args):
    if args.trigger_type == 'additive' and args.experiment == 'silent_killer':
        model_arch = args.model_arch
        entity = args.wandb_entity
        trigger_type = args.trigger_type
        experiment = args.experiment
        command = (f'python silent_killer.py -s {s} -t {t} --trigger_type {trigger_type} --eps 0.062745 '
                   f'--craft_model {model_arch} --eval_model {model_arch} '
                   f'--entity {entity} --name {trigger_type}-{experiment} --wandb true')
        if args.trigger_type == 'additive' and args.experiment == 'base':
            command = command + ' --n_samples 0 --crafting_repetitions 0'
        elif args.trigger_type == 'patch' and args.experiment == 'silent_killer':
            command = command + ' --trigger_opt_eps 1'
        elif args.trigger_type == 'patch' and args.experiment == 'base':
            command = command + ' --trigger_opt_eps 1 --trigger_opt false --trigger_init_method from_file'
    return command + '\n'


if __name__ == '__main__':
    main()
