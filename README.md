# OCN
### This is an implementation of OCN: One Core Neuron for time-series predicting and deep learning

## Getting Started
### Prediciton
1. Install requirements. `pip install -r requirements.txt`
2. For prediciton, download the [datasets](https://drive.google.com/file/d/1GhkRjq-p7EVM9y6JSHyG2DJ73JgJ0E7t/view?usp=drive_link) from Google drive and unzip it to the folder `./data`. 
3. The configaration files are in the folder `./cfg` and the script files for experiments are in the folder `./exp`. You can run the script `./exp/prediction_main.py` for training and validating the one-core-neuron system (**OCNS**) through the following command:
    ```
    cd exp && python ./prediction_main.py
    ```
### Classification
1. Follow the requirements installation step (first step) above.
2. Download datasets to folder `./data` from the appropriate source:
    |  Dataset   | Source  |
    |  ----  | ----  |
    | MNIST  | http://yann.lecun.com/exdb/mnist/ |
    | Fashion-MNIST | https://github.com/zalandoresearch/fashion-mnist |
    | CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |
    | CIFAR-100 (coarse label) | https://www.cs.toronto.edu/~kriz/cifar.html |
    | Cropped version of SVHN | http://ufldl.stanford.edu/housenumbers/ |
3. The configaration files are in the folder `./cfg` and the script files for experiments are in the folder `./exp`. You can run the script `./exp/classification_main.py` for training and validating the one-core-neuron system (**OCNS**) through the following command:
    ```
    cd exp && python ./classification_main.py
    ```

## Contact
If you have any questions or concerns, please contact us or submit an issue.
> - The co-first authors: Hao Peng (mahp_scut@mail.scut.edu.cn), Pei Chen (chenpei@scut.edu.cn);
> - The corresponding authors: Rui Liu (scliurui@scut.edu.cn), Luonan Chen (lnchen@sibs.ac.cn).