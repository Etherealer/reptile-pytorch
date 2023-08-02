import argparse

__all__ = ['argument_parser', 'optimizer_kwargs', 'train_kwargs', 'evaluate_kwargs']


def argument_parser() -> argparse.ArgumentParser:
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--pretrained', type=str, default=None, help='evaluate a pre-trained model path')
    parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint to save or load')

    parser.add_argument('--classes', type=int, default=5, help='number of classes per inner task')
    parser.add_argument('--shots', type=int, default=5, help='number of examples per class')
    parser.add_argument('--train-shots', type=int, default=0, help='shots in a training batch')
    parser.add_argument('--inner-batch', type=int, default=5, help='inner batch size')
    parser.add_argument('--inner-iters', type=int, default=20, help='inner iterations')
    parser.add_argument('--replacement', action='store_true', help='sample with replacement')
    parser.add_argument('--meta-step', type=float, default=0.1, help='meta-training step size')
    parser.add_argument('--meta-step-final', type=float, default=0.1, help='meta-training step size by the end')
    parser.add_argument('--meta-batch', type=int, default=1, help='meta-training batch size')
    parser.add_argument('--meta-iters', type=int, default=400000, help='meta-training iterations')
    parser.add_argument('--eval-batch', type=int, default=5, help='eval inner batch size')
    parser.add_argument('--eval-iters', type=int, default=50, help='eval inner iterations')
    parser.add_argument('--eval-samples', type=int, default=10000, help='evaluation samples')
    parser.add_argument('--eval-interval', type=int, default=10, help='train steps per eval')
    parser.add_argument('--transductive', action='store_true', help='evaluate all samples at once')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer name')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='optimizer step size')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay rate')
    return parser


def optimizer_kwargs(args: argparse.Namespace) -> dict:
    return {
        'name': args.optimizer,
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay
    }


def train_kwargs(args: argparse.Namespace) -> dict:
    return {
        'num_classes': args.classes,
        'num_shots': args.shots,
        'train_shots': (args.train_shots or None),
        'inner_batch_size': args.inner_batch,
        'inner_iters': args.inner_iters,
        'replacement': args.replacement,
        'meta_step_size': args.meta_step,
        'meta_step_size_final': args.meta_step_final,
        'meta_batch_size': args.meta_batch,
        'meta_iters': args.meta_iters,
        'eval_inner_batch_size': args.eval_batch,
        'eval_inner_iters': args.eval_iters,
        'eval_interval': args.eval_interval,
        'transductive': args.transductive,
    }


def evaluate_kwargs(args: argparse.Namespace) -> dict:
    return {
        'num_classes': args.classes,
        'num_shots': args.shots,
        'eval_inner_batch_size': args.eval_batch,
        'eval_inner_iters': args.eval_iters,
        'replacement': args.replacement,
        'num_samples': args.eval_samples,
        'transductive': args.transductive,
    }
