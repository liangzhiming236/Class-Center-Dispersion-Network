# Class Center Dispersion Network for Out-of-Distribution Detection

This repository contains the code and trained models for our paper Class Center Dispersion Network for Out-of-Distribution Detection.

### Requirements

- numpy
- sklearn
- pytorch

### Experimental data

- MNIST
- Fashion-MNIST
- E-MNIST-letters
- Not-MNIST
- Omniglot

### Usage

- Converting the dataset to png format:

  ```
  python convert.py
  ```

- Training a classification model without considering OOD samples:

  ```shell
  python pre_train.py
  ```

- Training and test a new model:

  ```shell
  python main.py
  ```

### Citation

- If you found this code useful please cite our paper. Thank you!

