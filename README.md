# SRDGI：Real-time Global Illumination via deep learning

This is the SRDGI project code for the cs337 course. This project proposes a method to further improve the performance of the network, balancing rendering quality with prediction time, to obtain a new network SRDGI, while regenerating a more diverse dataset, using which the SRDGI is trained with further enhanced generalization capabilities to meet a wider range of hardware and latency requirements.

![image-20220104184758796](CG.png)

# Dataset

we used Training images dataset (download)(https://jbox.sjtu.edu.cn/l/t1cKmu).

## Usage

### SRDGI

Train model

```
python train.py --dataset PATH_TO_DATASETDIR --n_epoch 50 --resume_G PATH_TO_RESUME_GENERATER --resume_D PATH_TO_RESUME_DISCRIMINATOR
```

- `--dataset`: path to dataset.
- `--n_epoch`: number of training epochs.
- `--resume_G`: path to generator checkpoints to resume.
- `--resume_D`: path to discriminator checkpoint to resume.

test model

```
python test_compare.py --dataset PATH_TO_DATASETDIR --model PATH_TO_MODEL
```

- `--dataset`: path to dataset.
- `--model`: path to model.

### Dynamic Channels

Before you train or test the dynamic channels model, you should prepare the dataset first and 
'''
cd Dynamic Channels/
'''

Train model

```
python train.py --dataset PATH_TO_DATASETDIR --n_epoch 50 --dynamic_channels 1
```

- `--dataset`: path to dataset.
- `--n_epoch`: number of training epochs.
- `--dynamic_channels`: set it to 1 for running with dynamic channel, 0 for full channel.

Render image

```
python test.py --model PATH_TO_MODEL --dataset PATH_TO_DATASETDIR --accuracy 1
```

- `--model`: path to model.
- `--dataset`: path to dataset.
- `--accuracy`: an integer between 1 and 4, the larger the rendering accuracy the higher.
