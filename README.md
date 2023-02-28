# Bachelor Thesis: Luis Denninger
Title: "Learning Implicit Probability Distribution Functions for 6D Object Poses" \
Author: Luis Denninger \
Supervisor: Arul Selvam Periyasamy \
First Examiner: Prof. Dr. Sven Behnke \
Second Examiner: Prof. Dr. Florian Bernard \

Description:\
This thesis proposes the ImplicitPosePDF model which aims to model pose distributions over all rigid body transformation $\mathbf{SE}(3)$ to capture object symmetries.
The estimation of the distributions is decoupled in two models estimating an orientation and a translation distribution respectively.
Each distribution is parametrized through a neural network. Using an efficient equi-volumetric sampling strategy for the rotation manifold $\mathbf{SO}(3)$ and the translation space $\mathbb{R}^3$, the pose distribution is approximated as a histogram over the respective space. The models are trained to estimate the likelihood of a single orientation, respectively translation, hypothesis taken from a set of ground-truth poses representing the object symmetries.
Furthermore, this thesis proposes the Automatic Pose Labeling Scheme to generate multiple pseudo ground-truth poses for each image corresponding to the symmetries to train the ImplicitPosePDF model.
Given an RGB-D image and a 3D object model, the pipeline produces the set of pseudo ground-truth poses through a two-stage point cloud registration process with a succeeding render-and-compare validation stage.

## Running Guide
The project can be run within a singularity image. The  singularity image can be found in the shared filesystem under: <pre> /home/nfs/inf6/data/singularity_images/rnc.sif </pre> \

