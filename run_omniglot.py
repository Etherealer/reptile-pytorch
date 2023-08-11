import torch
from torch import nn

from utils import setup_seed
from reptile import argument_parser, split_dataset, Omniglot, OmniglotModel, optimizers, optimizer_kwargs, \
    train, train_kwargs, evaluate, evaluate_kwargs

DATA_DIR = 'data'


def main():

    args = argument_parser().parse_args()
    setup_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_set, test_set = split_dataset(Omniglot(DATA_DIR, download=True))

    model = OmniglotModel(args.classes).to(device)
    optimizer = optimizers.make(model.parameters(), **optimizer_kwargs(args))
    criterion = nn.CrossEntropyLoss()
    if not args.pretrained:
        train(model, optimizer, criterion, device, train_set, test_set, args.checkpoint, **train_kwargs(args))
    else:
        model.load_state_dict(torch.load(args.pretrained))

    print('Evaluating dataset ......')
    print('train set acc: ' + str(evaluate(model, optimizer, criterion, device, train_set, **evaluate_kwargs(args))))
    print('test set acc: ' + str(evaluate(model, optimizer, criterion, device, test_set, **evaluate_kwargs(args))))


if __name__ == '__main__':
    main()
