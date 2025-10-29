# U-Net(Convolutional Networks for Biomedical Image Segmentation)

## Environment Configuration：
* Python3.6/3.7/3.8
* Pytorch1.10
* Ubuntu or Centos(Windows does not currently support multi-GPU training)
* It is best to train using a GPU.
* For detailed environment setup, see`requirements.txt`

## File Structure：
```
  ├── src: Code for building a U-Net model
  ├── train_utils: Modules related to training, validation, and multi-GPU training
  ├── train.py: Training using a single GPU as an example
  ├── train_multi_GPU.py: For users who use multiple GPUs
  ├── predict.py: A simple prediction script that uses trained weights for prediction testing
  └── compute_mean_std.py: Calculate the mean and standard deviation of each channel in the dataset
```


## Training methods
* Ensure the dataset is prepared in advance
* To train using a single GPU or CPU, directly use the train.py training script.
* To train using multiple GPUs, use the command `torchrun --nproc_per_node=8 train_multi_GPU.py`. The `nproc_per_node` parameter specifies the number of GPUs to use.
* If you want to specify which GPU devices to use, you can add `CUDA_VISIBLE_DEVICES=0,3` before the command (for example, if I only want to use the first and fourth GPU devices).
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## Precautions
* When using the training script, be sure to set `--data-path` to the **root directory** where your `DRIVE` folder is stored.
* When using the prediction script, set `weights_path` to the path of the weights you generated.
* When using the validation file, make sure that your validation or test set contains at least one example of each class. When using it, you only need to modify `--num-classes`, `--data-path`, and `--weights`; try not to change other parts of the code.
* All datasets are in the master branch
* Segmentation refers to segmenting the dataset.
* classification is a dataset for disease identification
