# Attentive Implicit Representation Networks (AIR-Nets)
[Preprint](https://github.com/SimonGiebenhain/AIR-Nets/tree/main/AIR-Nets.pdf) | [Supplementary](https://github.com/SimonGiebenhain/AIR-Nets/tree/main/AIR-Nets_supp.pdf) | Accepted at the International Conference on 3D Vision (3DV)

https://user-images.githubusercontent.com/28050396/138076268-b82b0de0-e148-46e8-a427-ac1016a339bb.mov


This repository is the offical implementation of the paper   
>**AIR-Nets: An Attention-Based Framework for Locally Conditioned Implicit Representations**  
>by *Simon Giebenhain* and *Bastian Goldluecke*

Furthermore it provides a unified framework to execute [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks) (ONets), [Convolutional Occuapncy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks) (ConvONets) and [IF-Nets](https://github.com/jchibane/if-net).  


More qualitative results of our method can be found [here](https://drive.google.com/drive/folders/1MfA292bnXIHhe8uvrt308tv1uEdrhwHU?usp=sharing).


# Install
All experiments with AIR-Nets were run using CUDA version 11.2 and the official pytorch docker image `nvcr.io/nvidia/pytorch:20.11-py3`, as published by nvidia [here](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch).
However, as the model is solely based on simple, common mechanisms, older CUDA and pytorch versions should also work.
We provide the `air-net_env.yaml` file that holds all python requirements for this project. To conveniently install them automatically with [anaconda](https://www.anaconda.com/) you can use:
```
conda env create -f air-net_env.yml
conda activate air-net
```

AIR-Nets use farthest point sampling (FPS) to downsample the input. Run
```
pip install pointnet2_ops_lib/.
```
inorder to install the cuda implementation of FPS. Credits for this go to Erik Wijams's [GitHub](https://github.com/erikwijmans/Pointnet2_PyTorch), from where the code was copied for convenience.

Running
```
python setup.py build_ext --inplace
```
installs the MISE algorithm (see http://www.cvlibs.net/publications/Mescheder2019CVPR.pdf) for extracting the reconstructed shapes as meshes.

When you want to run Convolutional Occupancy Networks you will have to install `torch scatter` using the official instructions found [here](https://github.com/rusty1s/pytorch_scatter).

# Data Preparation
In our paper we mainly did experiments with the [ShapeNet](https://shapenet.org/) dataset, but preprocessed in two different falvours. The following describes the preprocessing for both alternatives. Note that they work individually, hence there is no need to prepare both. (When wanting to train with noise I would recommend the Onet data, since the supervision of the IF-Net data is concentrated so close to the boundary that the problem gets a bit ill-posed (adapting noise level and supervision distance can solve this, however).)


## Prepating the IF-Net data

> A small disclaimer: Preparing the data as in this tutorial will produce ~700GB of data. Deleting the `.obj` and `.off` files should reduce the load to 250GB. Storage demand can further be reduced by reducing the number of samples in `data_processing/boundary_sampling.py`. If storage is scarce the ONet data (see below) is an alternative.

This data preparation pipeline is mainly copied from [IF-Nets](https://github.com/jchibane/if-net), but slightly simplified.

Install a small library needed for the preprocessing using
```
cd data_processing/libmesh/
python setup.py build_ext --inplace
cd ../..
```
Furthermore you might need to install `meshlab` and `xvfb` using
```
apt-get update
apt-get install meshlab
apt-get install xvfb
```

To install gcc you can run `sudo apt install build-essential`.


To get started, download the preprocessed data by [Xu et. al. NeurIPS'19] from [Google Drive](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr)
into the `shapenet` folder.

Then extract the files into `shapenet\data` using:

```
ls shapenet/*.tar.gz |xargs -n1 -i tar -xf {} -C shapenet/data/
```

Next, the input and supervision data is prepared.
First, the data is converted to the .off-format and scaled (such that the longest edge of the bounding box for each object has unit length) using
```
python data_processing/convert_to_scaled_off.py
```
Then the point cloud input data can be created using
```
python data_processing/sample_surface.py
```
which samples 30.000 point uniformly distributed on the surface of the ground truth mesh. During training and testing the input point clouds will be randomly subsampled from these surface samples.
The coordinates and corresponding ground truth occupancy values used for supervision during training can be generated using
```
python data_processing/boundary_sampling.py -sigma 0.1
python data_processing/boundary_sampling.py -sigma 0.01
```
where `-sigma` specifies the standard deviation of the normally distributed displacements added onto surface samples. Each call will generate 100.000 samples near the object's surface for which ground truth occupancy values are generated using the implicit waterproofing algorithm from [IF-Nets supplementary](http://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet_supp.pdf). I have not experimented with any other values for sigma, and just copied the proposed values.

In order to remove meshes that could not be preprocessed correctly (should not be more than around 15 meshes) you should run
```
python data_processing/filter_corrupted.py -file 'surface_30000_samples.npy' -delete
```
Pay attantion with this command, i.e. the directory of all objects that don't contain the `surface_30000_samples.npy` file are deleted. If you chose to use a different number points, please make sure to adapt the command accordingly.

Finally the data should be located in `shapenet/data`.

## Preparing the data used in ONets and ConvONets

To parapre the ONet data clone their [repository](https://github.com/autonomousvision/occupancy_networks). Navigate to their repo `cd occupancy_networks` and run

```
bash scripts/download_data.sh

```
which will download and unpack the data automatically (consuming 73.4 GB). 
From the perspective of the main repository this will place the data in `occupancy_networks/data/ShapeNet`.

## Preparing the FAUST dataset
In order to download the FAUST dataset visit http://faust.is.tue.mpg.de and sign-up there.
Once your account is approved you can download a `.zip`-file nameed `MPI-FAUST.zip`. Please place the extracted folder in the main folder, such that the data can be found in `MPI-FAUST`.


# Training

For the training and model specification I use `.yaml` files. Their structure is explained separately [here](https://github.com/SimonGiebenhain/AIR-Nets/configs/README.md).

To train the model run
```
python train.py -exp_name YOUR_EXP_NAME -cfg_file configs/YOUR_CFG_FILE -data_type YOUR_DATA_TYPE
```
which stores results in `experiments/YOUR_EXP_NAME`. `-cfg_file` specifies the path to the config file. The content of the config file will then also be stored in `experiments/config.yaml`. `YOUR_DATA_TYPE` can either be `'ifnet'`, `'onet'` or `'human'` and dictates which dataset to use.
Make sure to adapt the `batch_size` parameter in the config file accoridng to your GPU size.

Training progress is saved using tensorboard. You can visualize it by running
```
tensorboard --logdir experiments/YOUR_EXP_NAME/summary/ 
```

Note that checkpoints (including the optimizer) are saved after each epoch in the `checkpoints` folder. Therefore training can seamlessly be continued.


# Generation
To generate reconstructions of the test set, run
```
python generate.py -exp_name YOUR_EXP_NAME -checkpoint CKPT_NUM -batch_points 400000 -method REC_METHOD 
```
where `CKPT_NUM` specifies the epoch to load the model from and `-batch_points` specifies how many points are batched together and may have top be adapted to your GPU size.  
`REC_METHOD` can either be `mise` or `mcubes`. The former (and recommended) option uses the MISE algorithm for reconstruciton. The latter uses the vanilla marching cubes algorithm. For the MISE you can specifiy to additional paramters `-mise_res` (initial resolution, default is 64) and `-mise_steps` (number of refinement steps, defualt 2). (Note that we used 3 refinement steps for the main results of the dense models in the paper, just to be on the save side and not miss any details.) For the regular marching cubes algorithm you can use `-mcubes_res` to specify the resolution of the grid (default 128). Note that the cubic scaling quickly renders this really slow.

The command will place the generate meshes in the `.OFF`format in `experiments/YOUR_EXP_NAME/evaluation_CKPT_NUM_@mise_resxmise_steps/generation` or `experiments/YOUR_EXP_NAME/evaluation_CKPT_NUM_@mcubes_res/generation` depending on `method`.


# Evaluation
Running
```
python data_processing/evaluate.py -reconst -generation_path experiments/YOUR_EXP_NAME/evaluation_CKPT.../generation
```
will evaluate the generated meshes using the most common metrics: the volumetric IOU, the Chamfer distance (L1 and L2), the Normal consistency and F-score. 

The results are summarized in
`experiment/YOUR_EXP_NAME/evaluation_CKPT.../evaluation_results.pkl` by running
```
python data_processing/evaluate_gather.py -generation_path experiments/YOUR_EXP_NAME/evaluation_CKPT.../generation
```


# Pretrained Models

Weights of trained models can be found [here](https://drive.google.com/drive/folders/1_WiBfpVf8k-DvbOzy1QY34qM4uC2sfEJ?usp=sharing). For example create a folder `experiments/PRETRAINED_MODEL`, placing the corresponding config file in `experiments/PRETRAINED_MODEL/configs.yaml` and the weights in `experiments/PRETRAINED_MODEL/checkpoints/ckpt.tar`. Then run 

```
python generate.py -exp_name PRETRAINED_MODEL -ckpt_name ckpt.tar -data_type DATA_TYPE
```

# Contact
For questions, comments and to discuss ideas please contact Simon Giebenhain via simon.giebenhain (at] uni-konstanz {dot| de.


# Citation
```
@inproceedings{giebenhain2021airnets,
title={AIR-Nets: An Attention-Based Framework for Locally Conditioned Implicit Representations},
author={Giebenhain, Simon and Goldluecke, Bastian},
booktitle={2021 International Conference on 3D Vision (3DV)},
year={2021},
organization={IEEE}
}
```

# Acknowledgements
Large parts of this repository as well as the structure are copied from Julian Chibane's [GitHub repository](https://github.com/jchibane/if-net) of the [IF-Net](https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf) paper. Please consider also citing their work, when using this repository!

This project also uses libraries form [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks) by [Mescheder et al. CVPR'19](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks) and from [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks) by [Peng et al. ECCV'20].  
 We also want to thank [DISN](https://github.com/Xharlie/DISN) by [Xu et. al. NeurIPS'19], who provided their preprocessed ShapeNet data publicly.
 Please consider to cite them if you use our code.



# License
Copyright (c) 2020 Julian Chibane, Max-Planck-Gesellschaft and  
              2021 Simon Giebenhain, Universität Konstanz

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes. For commercial inquiries, please see above contact information.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the `Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion` paper and the `AIR-Nets: An Attention-Based Framework for Locally Conditioned Implicit Representations` paper in documents and papers that report on research using this Software.
