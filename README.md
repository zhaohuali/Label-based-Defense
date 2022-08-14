# Gradient-Heterogeneity
Reveal the “Leaker” and Block the Attacker: Defending Gradient Inversion Attacks on Federated Learning

## Abstract:
Federated learning is widely studied to protect local privacy by exchanging model parameters rather than raw data among participants. However, the community has found a method called Gradient Inversion Attack (i.e. GIA), which makes an emissary participant or parameter server of Federated learning infer the local data of other participants only based on the model parameters exchanged. In the framework and process of Federated learning, what features provide heuristic information for spying inference raw data, and how to defend against GIA? So far, this is still an urgent question for the academic community to answer. We define the concept of gradient heterogeneity (i.e. GH),  and provide a fundamental theoretical understanding of the defense feasibility by increasing the GH (eg. using asynchronous BatchNorm, local multi-round iterations) and the sample label repetition rate. We designed and implemented enough experiments with different perspectives to verify the effectiveness of the concept and method. Experimental results show that GIA fails to attack (i.e. without leaking any valid information about local raw data) during the whole training process of a deep convolutional network without losing the test accuracy of the model. Our approach is effective and practical, which will help support the privacy protection of large model construction in federated learning.

## Requirements
```
ray=1.9.1
pytorch=1.10.1
torchvision=0.11.2
matplotlib=3.5.0
prettytable=2.5.0
apex: https://github.com/NVIDIA/apex
```

## Model (ResNet-18) training under different defense schemes
The specific content of the defense scheme involved can be viewed in the paper
```
python training/main_pytorch.py --scheme [defense schemes] --data [imagenet-folder with train] --result-root [path of checkpoints]
```
where `--scheme [defense schemes]` represents the selected defense scheme, the optional schemes are: `pure` `RA` `DP-SGD` `GH`. `--result-root [path of checkpoints]` is the path to store the trained model parameters. For example, we chose the scheme GH of our paper and placed the trained model in the test folder:
```
python training/main_pytorch.py --scheme GH --data /data/imagenet/train --result-root /data/test
```

## Get shared gradient
By obtaining the corresponding gradient on the specified model parameters and settings, this gradient can be used in the following GIA.

### Gradients are computed by a single GPU
Assume that the GPU with serial number 0 is used for gradient calculation
```
python training/get_gradients.py --gpu 0 [training settings] --data [imagenet-folder with train] --results [path of results] --pretrained [checkpoint of trained model (.tar) or parameters of model (.pth)]
```
Among them, `[training settings]` can input specific gradient calculation settings, which involves the choice of defense means. The specific settings can be seen below. `--pretrained [checkpoint of trained model (.tar) or parameters of model (.pth)]` indicates the path of the trained model parameters(.pth) or checkpoint(.tar).

### Gradients are computed by multiple GPUs
The number of available GPUs can be adjusted by setting `os.environ["CUDA_VISIBLE_DEVICES"]` in the file `training/get_gradients.py`. We use 4 GPUs by default to compute gradients.
```
python training/get_gradients.py [training settings] --data [imagenet-folder with train] --results [path of results] --pretrained [checkpoint of trained model (.tar) or parameters of model (.pth)] --multiprocessing-distributed --dist-url tcp://127.0.0.1:10023 --dist-backend nccl  --world-size 1 --rank 0
```

### Selectable training settings
The contents that can be filled in `[training settings]` are listed in detail here.
1. GA-based defense scheme: `--kernel-size-of-maxpool 19 --ra`
2. Add noise to the gradient (DP-SGD): `--enable-dp --sigma 0.01 --max-per-sample-grad_norm 1 --delta 1e-5`
3. Use synchronous BatchNorm (default uses asynchronous BatchNorm): `--syncbn`. Only makes sense in a multi-GPU environment
4. Set the number of local iterations: `--epochs [the number of local iterations]`, A single iteration is `--epochs 1`
5. Simulated duplicate labels (each label has 4 duplicates, batch size must be 32): `--duplicate-label `
6. Set batch size (default is 32): `-b [batch size]`. set 79: `-b 79`
7. Dropout: We use the model VGG11 to test the impact of Dropout. The function of Dropout is enabled by default. If you want to close it, you need to add `--model-eval` to the command line. For example, get the gradient when Dropout is inactive: `--arch vgg11 --model-eval`

## Gradient Inversion Attack (GIA)

### Single GPU
```
python main_run.py --gpu 0 --checkpoint [path of the gradients(.tar)]  --min-grads-loss --metric
```
### Multiple GPUs
The number of available GPUs can be adjusted by setting `os.environ["CUDA_VISIBLE_DEVICES"]` in the file `main_run.py`. We use 4 GPUs by default to compute gradients.
```
python main_run.py --gpu 0 --checkpoint [path of the gradients(.tar)]  --min-grads-loss --metric --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:10036 --dist-backend nccl --multiprocessing-distributed
```