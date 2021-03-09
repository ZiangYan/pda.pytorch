# pda.pytorch

Code for our ICLR 2021 paper [Policy-Driven Attack: Learning to Query for Hard-label Black-box Adversarial Examples](https://openreview.net/forum?id=pzpytjk3Xb2).

## Environments
Our code is tested on the following environment (probably also works on other environments without many changes):

* Ubuntu 16.04
* Python 3.5.2
* CUDA 9.0.176
* CUDNN 7501
* PyTorch 1.1.0
* torchvision 0.2.0
* numpy 1.16.0 
* dill 0.3.2

## Datasets, Victim Models, and Pre-trained Policy Networks

For the required dataset, victim models, and pre-trained weights of policy networks, please download a tarball at [this link](https://drive.google.com/file/d/1dvxCYiv5GGM8Zl-7b-MEm0G3vard69lE/view?usp=sharing) and extract it.

After extraction, the ```data``` directory should include dataset and victim models for MNIST and CIFAR-10.

For ImageNet dataset, we use the same data format as in the [Pytorch official ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet), please put the dataset in ```data/imagenet```.

As described in our paper, we also use [ImageNetV2](https://github.com/modestyachts/ImageNetV2) to provide additional images for ImageNet experiments, please download the ImageNetV2 dataset and then use ```prepare_imagenet_valv2.py``` to merge ImageNetV2 with ImageNet.

The tarball also provides initial adversarial examples for targeted attacks (in ```data``` directory), and pre-trained policy networks (in ```output``` directory).

The structure of ```data``` directory should be:


```
data
├── cifar10
│   └── cifar-10-batches-py
├── cifar10-models
│   ├── carlinet
│   ├── madry_resnet50_l2_1_0
│   └── wrn_28_10_drop
├── imagenet 
│   ├── train
│   ├── val
│   └── valv2
├── init-adv-images-cifar10-test-carlinet.pth
├── init-adv-images-cifar10-test-madry_resnet50_l2_1_0.pth
├── init-adv-images-cifar10-test-wrn_28_10_drop.pth
├── init-adv-images-imagenet-valv2-resnet18.pth
├── init-adv-images-mnist-test-carlinet.pth
├── mnist
│   ├── processed
│   └── raw
└── mnist-models
    └── carlinet

```


## Usage

We provide off-the-shelf shell scripts to run attacks and reproduce the results in our paper.

For example, the following command will reproduce the MNIST->CNN->Ours row in Table 1 of our paper:

```
./pda_mnist_carlinet_untargeted_l2.sh
```

## Acknowledgements
The following resources are very helpful for our work:

* [Pretrained models and for ImageNet](https://github.com/Cadene/pretrained-models.pytorch)
* [Pretrained models for CIFAR-10](https://github.com/bearpaw/pytorch-classification)
* [Carlini's CIFAR-10 ConvNet](https://github.com/carlini/nn_robust_attacks)
* [Pretrained ConvNet models from AutoZoom](https://github.com/IBM/Autozoom-Attack)

## Citation
Please cite our work in your publications if it helps your research:

```
@inproceedings{yan2021policy,
  title={Policy-Driven Attack: Learning to Query for Hard-label Black-box Adversarial Examples},
  author={Yan, Ziang and Guo, Yiwen and Liang, Jian and Zhang, Changshui},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

