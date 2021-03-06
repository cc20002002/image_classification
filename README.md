﻿# COMP5328 - Advanced Machine Learning

## Assignment 2 - Learning with Noisy Data

**Author** : Yutong Cao,  Chen Chen,  Yixiong Fang

**Lecturer**: Tongliang Liu

**Tutors** : Nicholas James, Zhuozhuo Tu, Liu Liu

**Objectives:**
This project implements three algorithms based on support vector machine against class-dependent classification noise.

The first algorithm extends an existing Expectation Maximisation method to class-dependent classification noise. The method was based on: Expectation Maximisation by [Biggio et al. [2011]](http://proceedings.mlr.press/v20/biggio11/biggio11.pdf)

The second algorithm reproduces existing state-of-the-art algorithms which are robust to label noise. The algorithm is based on: Importance Reweighing by [Liu and Tao [2016]](https://arxiv.org/pdf/1411.7718.pdf)

We also heuristically  propose a `quick and dirty' approach, which we called:
3. Heuristic Approach by Relabelling

This code will compare the performance of these three algorithms on two well-known dataset:
 1. MINIST
 2. CIFAR

Both datasets are injected with label noise.

**Reuirements:**
- sklearn == 0.20.0
- multiprocessing
- numpy
- matplotlib
- densratio


Running Environment Setup
------------

1. Make sure to put the dataset file `mnist_dataset.npz` and `cifar_dataset.npz` into the folder with name `input_data` under the `Code` directory. The tree structure of our algorithm is:

```
project
│   README.md
│   biggio11.pdf
│   ...
└───Code
│   └───algorithm
│   │    │   main.py
│   │    │   util.py
│   │    │   estimate_rho_PCA.py
│   └───input_data
│        │   mnist_dataset.npz
         │   cifar_dataset.npz
└───assignment2
    │   ...
    │   ...
```

2. Run `main.py` in `Code/algorithm` with the choose dataset and algorithm.
   ```
    --dset DSET      Set the dataset to use, 1 = MINIST, 2 = CIFAR. Default is
                   CIFAR.
    --method METHOD  Set the algorithm to run, 1 = Expectation Maximisation, 2 =
                   Importance Reweighting, 3 = Heuristic Approach. Default is
                   'Importance Reweighting'.
   ```            

   For example, to run `Expectation Maximisation` on **MINIST** dataset, please run:

   ```
   python main.py --dset=1 --method=1
   ```
   
   If you do not set the parameter, the default would be running `Importance Reweighting` on **CIFAR**.  

3. To run algorithm estimating flip rates rho, please run in `Code/algorithm`:

   ```
   python estimate_rho_PCA.py
   ```

All results will be auto-saved to `result/{generated-time-dataname}`. 
