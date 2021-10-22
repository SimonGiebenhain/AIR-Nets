# Model and Training Specification
The config files use the `.yaml` format in order to specify the experiment setting, traing hyperparameters and model configuration.
`airnet300.yaml` and `airnet3000.yaml` are the settings used in the main paper for the sparse and dense setting respectively. 

In the following the individual parts of the config file are explained. Since the vector attention mechanism used throughout AIR-Nets is quite memory intensive, we additionally include some hints on reducing the memory footprint in the end.



## Training 
An examplary `training` seciton looks as such:
```
training:
  lr: 0.0005 # learning rate
  lr_decay_interval: 200  #interval to reduce lr, null for constant lr
  lr_decay_factor: 0.2  #reduction factor
  grad_clip: null  #norm for gradient clipping, null for no clipping
  npoints_decoder: 1000  #number of points used to calculate the loss with
  ckpt: null  #epoch number from where to resume training
  batch_size: 64 
  pc_samples: 300  #number of input points
```
## Experiment Setting
So far `noisy_obs` is the only option for the experiemnt setting (which is called `mode`). `null` means no noise, otherwise it will be interpreted as the standard deviation of gaussian noise added to to the observed input point cloud. 
We followed ONets and ConvONets and used `0.005` with the ONet dataset.
Using noise together with the IF-Net dataset is not optimal as explained in the main `README.md`.

## Model Architecture
The model architecture specification is divided into the `encoder` and `decoder`.
The `type` filed specifies the type of model to be used. For the encoder it can be one of the following:
```
'airnet', 'ifnet', 'onet', 'convonet', 'pointnet++'.
```
while the options for the decoder are
```
'airnet', 'ifnet', 'onet', 'convonet', 'interp', 'ldif'.
```
Here `interp` and `ldif` refer to simple interpolation based decoders, as explained in our supplementary.
Furthermore note that `pointet++` specifies a Pointnet++-style encoder followed by a certain number of full-attenten VSA modules (as in AIR-Nets).

Note that the parameter options change based on the model type used, obviously AIR-Nets have different paramters as ONets or IF-Nets.
Therefore, we refer to repositories of the individual papers for an explanation of these parameters.
In the following we explain the parameters for AIR-Nets in more detail.

### Encoder

```
encoder:
  type: 'airnet'
  encoder_attn_dim: 256  #dim for transformer and all linear layers.
  npoints_per_layer: [300, 200, 100]  #cardinalities 
  encoder_nneigh: 16  #neighborhood size
  encoder_nneigh_reduced: 16 #neighborhood size of first layer
  nfinal_trans: 3  #number of full attention layers
  encoder_attn_dim_reduced: 256  #dim for first layer
```

### Decoder
The decoder specifications e.g. look like
```
decoder:
  type: 'airnet'
  decoder_attn_dim: 200  # dim for VCA in decoder
  decoder_nneigh: 7  # neighborhood size for VCA
  decoder_hidden_dim: 128  # dim of final FFN
  decoder_nblocks: 5  # num layers of final FFN
```

### Comments on Memory Demand

The vector attention mechanism is quite memroy intensive, since the attention weights are vector valued, compared to the common use of scalar-valued attention mechanisms. Note the we have not experimented with using scalar-valued attention mechanisms or other graph neural network architectures. This is subject of future work. For now we only state some parameters which have a strong impact on memory demand (see our supplementary for more information).

An example configuration is that requires siginifcantyl less memory while preserving most of the reconstruction quality is `airnet300_lessMemory.yaml`. It demonstartes relatively 'save' ways to reduce memory, such as setting `M=50`, or reducing `encoder_attn_dim_reduced` and `encoder_nneigh_reduced` (the latter two specify the parameters for the very first layer). Clearly reducing `encoder_attn_dim` also saves memory. For the decoder `decoder_nneigh` (valued depends on `M`) and `decoder_attn_dim` (we have not experimented with this parameter) also have a strong effect.

For example using at a `batch_size=64` using `airnet300_lessMemory.yaml` instead of `arinet300.yaml` reduced the GPU memory demand from ~36GB to ~18GB, while 'only' leading to a ~15% drop in validation loss.   
Please make sure to adapt the batch size to you GPU size.

