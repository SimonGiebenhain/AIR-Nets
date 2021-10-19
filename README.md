# AIR-Nets
Attentive Implicit Representation Networks (AIR-Nets)  

![](GIFS/teaser.gif)

More qualitative results:
[3D Reconstruction from 300 input points](https://github.com/SimonGiebenhain/AIR-Nets/tree/main/GIFS)  
[3D Reconstruction from 3000 input points](https://github.com/SimonGiebenhain/AIR-Nets/tree/main/GIFS_3000)


# Install
TODO
All experiments with AIR-Nets were run using CUDA version 11.2 and the official pytorch docker image `nvcr.io/nvidia/pytorch:20.11-py3`, as published by nvidia.
However, as the model is solely based on common mechanisms, older CUDA and pytorch versions should also work.
We provide the `air-net_env.yaml` file that holds all python requirements for this project. To conveniently install them automatically with [anaconda](https://www.anaconda.com/) you can use:
```
conda env create -f air-net_env.yml
conda activate air-net
```

Run
```
pip install pointnet2_ops_lib/.
```
inorder to install the cuda implementation of farthest point sampling (FPS).



# Data Preparation
TODO
The data preparation pipeline is copied from [IF-Nets](https://github.com/jchibane/if-net) and slightly adapted to prepare input point clouds without discretization. Furthermore only a subset of the prescribed steps has to be taken. Therefore the following repeats only the necessary steps.

To run the preparation processes please use the provided `if-net_env.yml` file, containing all necessary dependencies for the preprocessing. Afterwards, two small libraries have to be built using
```
cd data_processing/libmesh/
python setup.py build_ext --inplace
cd ../..
```
Note that the fully prepared data will consume a few hundred GB of storage and as the IF-Net authors note:

> This project uses libraries for [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks) by [Mescheder et. al. CVPR'19] 
> and the ShapeNet data preprocessed for [DISN](https://github.com/Xharlie/DISN) by [Xu et. al. NeurIPS'19], please also cite them if you use our code.


To get started, download the [ShapeNet](https://www.shapenet.org/) data preprocessed by [Xu et. al. NeurIPS'19] from [Google Drive](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr)
into the `shapenet` folder.

Then extract the files into `shapenet\data` using:

```
ls shapenet/*.tar.gz |xargs -n1 -i tar -xf {} -C shapenet/data/
```

Next, the input and supervision data is prepared.
First, the data is converted to the .off-format and scaled (such that the longest edge of the bounding box for each object has unit length) using
```
python preprocessing/convert_to_scaled_off.py
```
Then the point cloud input data can be created using
```
python preprocessing/sample_surface.py
```
which samples 30.000 point uniformly distributed on the surface of the ground truth mesh. During training and testing the input point clouds will be randomly subsampled from these surface samples.

The coordinates and corresponding ground truth occupancy values used as supervision during training can be generated using
```
python preprocessing/boundary_sampling.py -sigma 0.1
python preprocessing/boundary_sampling.py -sigma 0.01
```
where `-sigma` specifies the standard deviation of the normally distributed displacements added onto surface samples. Each call will generate 100.000 samples near the object's surface for which ground truth occupancy values are generated using the implicit waterproofing algorithm from [IF-Nets supplementary](http://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet_supp.pdf).


In order to remove meshes that could not be preprocessed (should not be more than around 15 meshes) you should run
```
python preprocessing/filter_corrupted.py -file 'surface_30000_samples.npy' -delete
```


# Training
To train the model run
```
python train.py -exp_name YOUR_EXP_NAME -pc_samples NUM_INPUT_POINTS
```
which stores results in `experiments/YOUR_EXP_NAME`. `-pc_samples` dicates how many points will be subsampled as input during training.
Make sure adapt the optional argument `-batch_size` to your GPU size. 

Checkpoints are saved every epoch and you can view the tensorboard logs using
```
tensorboard --logdir experiments/name_of_exp/summary/ --host 0.0.0.0
```

The model hyperparameters are specified using the `CFG` dictionary in the `train.py` file, that has to be manually modified. The most important, non-obvious hyperparameters are listed below:
+ `npoints_decoder`: number of points used as supervision, we used `1.000` for all experiments (IF-Nets used 50.000)
+ `encoder_attn_dim`: number of dimensions used throught the whole model (i.e. dims in linear layers and inside vector attention)
+ `npoints_per_layer`: specifies downsampling stages and number of downsampling layers, i.e. `[300, 200, 100]` was used for sparse setting, which means that that the input point with 300 points is downsampled to 200 and then downsampled to 100. `[3000, 500, 100]` was used for dense setting.
+ `encoder_nneigh`: The number of neighbors used for the local attention mechanism, as well as for the set abstraction module. `16` was used for all experiments
+ `enocder_nneigh_reduced`: The number of neighbors used for the very first transformer block. Was set to `16` for sparse, but to `10` for the dense setting, as computing local attention for 3000 points is quite expensive.
+  `encoder_attn_dim_reduced`: Number of dimensions used for first transformer block.
+ `nfinal_trans`: Number of final transformer blocks acting on `npoints_per_layer[-1]` points; was set to 3 for all experiments.
+ `full_SA`: Boolean; specifies whether to use full self attention; Always set to `True`.
+ `shift`: Boolean; specifies whether network should learn to predict translations for each point; only beneficial for specific data.
+ `decoder_attn_dim`: Number of dimensions to be used in cross vector attention in decoder.
+ `decoder_nneigh`: Number of neighbors for cross vectora attention.
+ `decoder_hidden_dim`: Number of dimensions to be used in simple feed-forward-network (FFN) for final occupancy prediction.
+ `decoder_nblocks`: Number of blocks in final FFN.

The config dictionary will be stored in `experiments/YOUR_EXP_NAME/configs.json` and loaded when continuing training.

# Generation
To generate the reconstructed meshes for the test set, run
```
python generate.py -exp_name YOUR_EXP_NAME -checkpoint CKPT_NUM -batch_points 400000 -pc_samples NUM_INPUT_POINTS -retrieval_res RES
```
This will place the generate meshes in the `.OFF`format in `experiments/YOUR_EXP_NAME/evaluation_CKPT_NUM_@RES/generation`. 
`-retrieval_res` specifies the resolution used for the marching cubes algorithm. Note that the runtime scales cubically w.r.t. `RES`. We used 256, which achieves visually pleasing results, but takes about 11-12s per mesh.
Also, consider using the maximal `-batch_points` possible for your GPU.

> The generation script can be run on multiple machines in parallel in order to increase generation speed significantly. 

# Evaluation
Running
```
python data_processing/evaluate.py -reconst -generation_path experiments/YOUR_EXP_NAME/evaluation_CKPT_NUM_@RES/generation
```
will evaluate the generated meshes using 3 metrics: the vIOU, the Chamfer distance and a Normal consistency score. 
> Again this script can be run in parallel on multiple machines.

The results are gathered in 
`experiment/YOUR_EXP_NAME/evaluation_CKPT_NUM_@RES` by running
```
python data_processing/evaluate_gather.py -generation_path experiments/YOUR_EXP_NAME/evaluation_CKPT_NUM_@RES/generation
```


# Pretrained Models
Not published yet, as the code has to be reworked and the model arhcitecture might still change slightly.

# Contact
simon.giebenhain (at] uni-konstanz {dot| de


# Citation
```
@inproceedings{giebenhain2020airnets,
title={AIR-Nets: An Attention-Based Framework for Locally Conditioned Implicit Representations},
author={Giebenhain, Simon and Goldluecke, Bastian},
booktitle={2021 International Conference on 3D Vision (3DV)},
year={2021},
organization={IEEE}
}
```

# Acknowledgements
A huge thanks to Julian Chibane, the author of IF-Nets and the corresponding [GitHub](https://github.com/jchibane/if-net) repository, that serves as a baseline for this repo. Please also cite their work!



# License
Copyright (c) 2020 Julian Chibane, Max-Planck-Gesellschaft and  
              2021 Simon Giebenhain, Universität Konstanz

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes. For commercial inquiries, please see above contact information.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion paper in documents and papers that report on research using this Software.
