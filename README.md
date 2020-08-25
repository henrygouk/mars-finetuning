Fine-Tuning with MARS Distance Constraints
==========================================

This repo contains the code used for conducting experiments in the paper [Distance-Based Regularisation of Deep Networks for Fine-Tuning](https://arxiv.org/abs/2002.08253).

The code uses the keras API bundled with tensorflow, and has been tested with the official TF 2.1 docker image. The `scripts/` directory contains bash scripts for reproducing the experiments in the paper.

Running an Experiment
---------------------

The main file of interest is `finetune.py`. This is a script that will fine-tune and test a model on a supplied dataset. E.g.,

```
python3 finetune.py --network=resnet101 --dataset=/path/to/flowers --reg-method=constraint --reg-norm=inf-op --reg-extractor=6.6 --reg-classifier=7.6 --test
```

where `/path/to/flowers` is the path to a dataset containing `train/`, `val/` and `test/` subdirectories, each of which contain images stored in the format expected by the keras `ImageDataGenerator.flow_from_directory` method. A copy of the VGG-Flowers dataset stored in this format can be downloaded from [here](https://www.henrygouk.com/flowers.tar.gz). One can expect this performance when training a ResNet-101 on the VGG-Flowers dataset:

| Method               | Accuracy   |
| -------------------- | ---------- |
| Standard Fine-Tuning |     76.68% |
| L2-SP                |     83.11% |
| DELTA                |     86.57% |
| MARS-PGM             | **87.42%** |


How it Works
------------

The MARS fine-tuning regulariser improves the performance of fine-tuned networks by limiting how much the weights of the neural network can be changed by stochastic gradient descent during the fine-tuning process. There are two important concepts involved:

* How does one measure the distance between the pre-trained weights and the fine-tuned weights?
* How can the distance be restricted during training?

We show both theoretically and empirically that good generalisation performance can be achieved with a distance metric based on the Maximum Absolute Row Sum (MARS) norm:

![||W-V||_{MARS}=\max_j\sum_i|W_{j,i}-V_{j,i}|](https://render.githubusercontent.com/render/math?math=\huge||W-V||_{MARS}=\max_j\sum_i|W_{j,i}-V_{j,i}|.)

The regularisation strategy we employ is to apply a hard constraint to the MARS distance between the pre-trained and fine-tuned weights in each layer. This is accmplished through the use of projected gradient descent---our paper explains in detail why this is a more appropriate strategy than adding a penalty term to the loss function.

Citation
--------

If you happen to use this code (or method) in an academic context, please cite the following paper

```
@article{gouk2020,
  title={Distance-Based Regularisation of Deep Networks for Fine-Tuning},
  author={Gouk, Henry and Hospedales, Timothy M and Pontil, Massimiliano},
  journal={arXiv preprint arXiv:2002.08253},
  year={2020}
}
```
