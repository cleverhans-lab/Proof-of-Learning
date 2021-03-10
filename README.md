# Proof-of-Learning

This repository is an implementation of the paper [Proof-of-Learning: Definitions and Practice](https://PLACEHOLDER), published in 42nd IEEE Symposium on
Security and Privacy. In this paper, we introduce the concept of proof-of-learning in ML. Inspired by research on both proof-of-work and verified computing, we observe how a seminal training algorithm, gradient descent, accumulates secret information due to its stochasticity. This produces a natural construction for a proof-of-learning which demonstrates that a party has expended the compute require to obtain a set of model parameters correctly. For more details, please read the paper.

We test our code on two datasets: CIFAR-10, and CIFAR-100. 

### Dependency
Our code is implemented and tested on PyTorch. Following packages are used:
```
numpy
pytorch==1.6.0
torchvision==0.7.0
scipy==1.6.0
```

### Train
To train a model and create a proof-of-learning:
```
python train.py --save-freq [checkpointing interval] --dataset [any dataset in torchvision] --model [models defined in model.py or any torchvision model]
```
`save-freq` is checkpointing interval, denoted by k in the paper. There are a few other arguments that you could find at the end of the script. 

Note that the proposed algorithm does not interact with the training process, so it could be applied to any kinds of gradient-descent based models.


### Verify
To verify a given proof-of-learning:
```
python verify.py --model-dir [path/to/the/proof] --dist [distance metric] --q [query budget] --delta [slack parameter]
```
Setting q to 0 or smaller will verify the whole proof, otherwise the top-q iterations for each epoch will be verified. More information about `q` and `delta` can be found in the paper. For `dist`, you could use one or more of `1`, `2`, `inf`, `cos` (if more than one, separate them by space). The first 3 are corresponding l_p norms, while `cos` is cosine distance. Note that if using more than one, the top-q iterations in terms of all distance metrics will be verified.

Please make sure `lr`, `batch-sizr`, `epochs`, `dataset`, `model`, and `save-freq` are consistent with what used in `train.py`.

### Questions or suggestions
If you have any questions or suggestions, feel free to raise an issue or send me an email at nickhengrui.jia@mail.utoronto.ca
