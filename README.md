# Reptile-Pytorch
This project is a PyTorch implementation of [Reptile](https://arxiv.org/abs/1803.02999), a Meta-Learning algorithm that focuses on learning an initial set of parameters. Reptile is similar to MAML (Model-Agnostic Meta-Learning) in that regard. It can be considered a novel first-order gradient-based Meta-Learning algorithm, bearing some resemblance to FOMAML (First Order Model-Agnostic Meta-Learning).

The purpose of this project is to recreate the implementation process of [supervised-reptile](https://github.com/openai/supervised-reptile) in TensorFlow as closely as possible, using PyTorch.
## Getting the data

You can download the dataset by running [data.sh](data.sh) or by setting the dataset download attribute to True. Both options will automatically download the dataset.
```shell
./data.sh 
```
```
Omniglot(DATA_DIR, download=True)
```

## Reproducing training runs

You can train models with the `run_omniglot.py` scripts. Hyper-parameters are specified as flags (see `--help` for a detailed list).

```shell
# transductive 1-shot 5-way Omniglot.
python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15t --transductive

# 5-shot 5-way Omniglot.
python -u run_omniglot.py --train-shots 10 --inner-batch 10 --inner-iters 5 --learning-rate 0.001 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --checkpoint ckpt_o55

# 1-shot 5-way Omniglot.
python -u run_omniglot.py --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15

# 1-shot 20-way Omniglot.
python -u run_omniglot.py --shots 1 --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o120

# 5-shot 20-way Omniglot.
python -u run_omniglot.py --classes 20 --inner-batch 20 --inner-iters 10 --meta-step 1 --meta-batch 5 --meta-iters 200000 --eval-batch 10 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o520
```

When you specify `--checkpoint`, if it is a file path, the training will resume from that checkpoint, and the subsequent checkpoints will also be saved in the same directory as the file. If it is a folder path, the subsequent training checkpoints will be saved in that path, and there will be no recovery operation.

To evaluate with transduction, pass the `--transductive` flag. In this implementation, transductive evaluation is faster than non-transductive evaluation since it makes better use of batches.

If you just want to validate the model instead of training a new one from scratch, you can specify the path to the saved model using `--pretrained`. During the training process, the model will be saved to the default output folder after training completion. Additionally, the output files for TensorBoard will also be stored in the same folder.

## Result

The experimental results are all based on the configurations provided in the paper and have not been modified.

| 5 way 1 shot Acc | 5 way 5 shot Acc |
|:----------------:|:----------------:|
|      95.77%      |      98.10%      |

## References

- [Original Paper](https://arxiv.org/abs/1803.02999): Alex Nichol, Joshua Achiam, John Schulman. "On First-Order Meta-Learning Algorithms".
- Original code in Tensorflow: https://github.com/openai/supervised-reptile