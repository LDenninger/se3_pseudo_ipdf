# Bachelor Thesis: Luis Denninger
Title: "Learning Implicit Probability Distribution Functions for 6D Object Poses" \
Author: Luis Denninger \
Supervisor: Arul Selvam Periyasamy \
First Examiner: Prof. Dr. Sven Behnke \
Second Examiner: Prof. Dr. Florian Bernard 

Description:\
This thesis proposes the ImplicitPosePDF model which aims to model pose distributions over all rigid body transformation $\mathbf{SE}(3)$ to capture object symmetries.
The estimation of the distributions is decoupled in two models estimating an orientation and a translation distribution respectively.
Each distribution is parametrized through a neural network. Using an efficient equi-volumetric sampling strategy for the rotation manifold $\mathbf{SO}(3)$ and the translation space $\mathbb{R}^3$, the pose distribution is approximated as a histogram over the respective space. The models are trained to estimate the likelihood of a single orientation, respectively translation, hypothesis taken from a set of ground-truth poses representing the object symmetries.
Furthermore, this thesis proposes the Automatic Pose Labeling Scheme to generate multiple pseudo ground-truth poses for each image corresponding to the symmetries to train the ImplicitPosePDF model.
Given an RGB-D image and a 3D object model, the pipeline produces the set of pseudo ground-truth poses through a two-stage point cloud registration process with a succeeding render-and-compare validation stage.

For further questions you can contact me under: `l_denninger@uni-bonn.de` or `Luis0512@web.de`

## Installation/Running Guide
The project can be simply cloned from the repository. The files can be run using the singularity image provided in the shared filesystem: `/home/nfs/inf6/data/singularity_images/rnc.sif` \
Unfortunately, the image is missing a few necessary packages. Thus, please make sure to run the the script `install_lib.sh`. 

Moreover, to be able to use the Stillleben library we require an older version of the PyTorch library that does not include the ConvNeXt feature extractor and the pre-trained weights. \
Thus, we implemented the ConvNeXt ourselves according to the PyTorch implementation and the pre-trained weights have to be manually loaded from the PyTorch Hub to: `./se3_ipdf/models/weights/{ConvNeXt_TIny_Weights.pth/ConvNeXt_Small_Weights.pth/ConvNeXt_Base_Weights.pth}` 

### Running

The files from this project can then be run using: <pre> singularity run --nv  --bind /usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d -B /home/nfs/inf6/data/datasets/ --env PATH=$HOME/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /home/nfs/inf6/data/singularity_images/rnc.sif python </pre>

To train the model and test the model, one needs to first initiate an experiment using `./initiate_exp.py`. The rotation and translation can be separately trained using `./train_model.py`.
To combine them to a single ImplicitPosePDF model, one needs to run `./make_ensamble_model.py`. \
For evaluation of the trained model use `./evaluate_model.py`. To produce visualizations produce `./visualize_model.py`

## Datasets
The model is evaluated on a custom dataset, the Photorealistic dataset and the T-Less dataset. The datasets can be found in the shared filesystem. \
Photorealistic dataset: `/home/nfs/inf6/data/datasets/IPDF_tabletop` \
T-Less dataset: `/home/nfs/inf6/data/datasets/T-Less` 

## Project Structure

### Configurations
The config files and utils functions for the config for the ImplicitPosePDF and Automatic Pose Labeling Scheme can be found under: `./config` \
The default configuration files are provided. Specific config files for the different datasets and objects can be generated manually using `./generate_config_file.py` 

### Data
The PyTorch dataloader for the different dataset and the functions to load the object models can be found under: `./data` 

### ImplicitPosePDF Model
The ImplicitPosePDF model and all corresponding methods can be found under: `./se3_ipdf` \
The Rotation-IPDF, Translation-IPDF, IPPDF and ConvNeXt models can be found under: `./se3_ipdf/models` \
The functions used for sampling the rotation and translation space can be found under: `./se3_ipdf/utils` \
The functions for evaluation can be found under: `./se3_ipdf/evaluation` 

### Automatic Pose Labeling Scheme
The complete pose labeling scheme can be found under: `./pose_labeling_scheme` \
The methods used in the three different stages can be found under: `./pose_labeling_scheme/registration` 

### Visualization
The functions used for visualization to produce the qualitative results can be found under: `./utils/visualizations`

## Thesis
A copy of the thesis and complementary material can be found under `./thesis` 



