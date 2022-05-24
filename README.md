# BPLC+NOSO
This repository is the official implementation of BPLC+NOSO: Backpropagation of errors based on latency code with neurons that only spike once at the most.
BPLC is backpropagation of errors based on latency code, which is mathematically rigorous given that no approximations of any gradient evaluations are used. When combined with neurons that spike once at the most (NOSOs), BPLC+NOSO highlights the following advantages of learning efficiency: (i) computational complexity for learning is independent of the input encoding length, and (ii) only few NOSOs are active during learning and inference periods, leading to large reduction in computational complexity. 

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

### Dataset
[MNIST dataset](http://yann.lecun.com/exdb/mnist/)

[N-MNIST](https://www.garrickorchard.com/datasets/n-mnist) dataset loader was adapted from [PySNN](https://github.com/BasBuller/PySNN) and modified.

## Training
To train SNNs (FMNISTnet and CIFARnet) using BPLC on Fashion-MNIST or CIFAR-10, run this command:
```train
cd BPLC+NOSO
python main.py --task <FMNIST or CIFAR10> --network <FMNISTnet or CIFARnet> --mode train
```

## Evaluation
To evaluate a folded SNNs( FMNISTnet and CIFARnet) on Fashion-MNIST or CIFAR-10, run this command:
```evaluation
cd BPLC+NOSO
python main.py --task <FMNIST or CIFAR10> --network <FMNISTnet or CIFARnet> --mode eval
```


## Results
Our model achieves the following performance on: 

| Method   | Network          | Dataset          | Accuracy (%) | # spikes (inference)  |
| -------- |------------------| -----------------|--------------|-----------------------|
| BPLC+NOSO| FMNISTnet        | Fashion-MNIST    | 92.15%       |  16K ± 0.48K          |
| BPTC+NOSO| CIFARnet         | CIFAR-10         | 87.19%       |  187K ± 0.39K         |

FMNISTnet : 32C5-P2-64C5-P2-600-10 \\
CIFARnet : 64C5-128C5-P2-256C5-P2-512C5-256C5-1024-512-10

