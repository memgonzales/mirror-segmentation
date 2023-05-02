# Designing a Lightweight Edge-Guided CNN for Segmenting Mirrors and Reflective Surfaces
![badge][badge-python]
![badge][badge-numpy]
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)  <br>
[![Actions Status](https://github.com/bioinfodlsu/phage-host-prediction/workflows/Check%20for%20syntax%20errors/badge.svg)](https://github.com/bioinfodlsu/phage-host-prediction/actions)
![badge][badge-github-actions]

**This work was accepted for full paper presentation at the 2023 International Conference in Central Europe on Computer Graphics, Visualization and Computer Vision ([WSCG 2023](http://www.wscg.eu/)), held virtually and in-person in Pilsen, Czech Republic:**

- Our preprint can be accessed via this [link](https://github.com/memgonzales/mirror-segmentation/blob/main/Designing%20a%20Lightweight%20Edge-Guided%20CNN%20for%20Segmenting%20Mirrors%20and%20Reflective%20Surfaces.pdf).
- Our [dataset of mirrors and reflective surfaces](https://github.com/memgonzales/mirror-segmentation#dataset) is publicly released for future researchers.

## Table of Contents
- [Description](https://github.com/memgonzales/mirror-segmentation#description)
- [Running the Model](https://github.com/memgonzales/mirror-segmentation#running-the-model)
  - [Training](https://github.com/memgonzales/mirror-segmentation#training)
  - [Prediction](https://github.com/memgonzales/mirror-segmentation#prediction)
  - [Evaluation](https://github.com/memgonzales/mirror-segmentation#evaluation)
  - [Models & Weights](https://github.com/memgonzales/mirror-segmentation#models--weights)
- [Dataset](https://github.com/memgonzales/mirror-segmentation#dataset)
- [Dependencies](https://github.com/memgonzales/mirror-segmentation#dependencies)
- [Attributions](https://github.com/memgonzales/mirror-segmentation#attributions)
- [Authors](https://github.com/memgonzales/mirror-segmentation#authors)

## Description
**ABSTRACT:** The detection of mirrors is a challenging task due to their lack of a distinguishing appearance and the visual similarity of reflections with their surroundings. While existing systems have achieved some success in mirror segmentation, the design of lightweight models remains unexplored, and datasets are mostly limited to clear mirrors in indoor scenes. In this paper, we propose a new dataset consisting of 454 images of outdoor mirrors and reflective surfaces. We also present a lightweight edge-guided convolutional neural network based on PMDNet. Our model uses EfficientNetV2-Medium as the backbone, and employs parallel convolutional layers and a lightweight convolutional block attention module to capture both low-level and high-level features for edge extraction. It registered $F_\beta$ scores of 0.8483, 0.8117, and 0.8388 on MSD, PMD, and our proposed dataset, respectively. Applying filter pruning via geometric median resulted in $F_\beta$ scores of 0.8498, 0.7902, and 0.8456, respectively, performing competitively with the state-of-the-art PMDNet but with 78.20&times; fewer FLOPS and 238.16&times; fewer parameters.

**INDEX TERMS:** Mirror segmentation, Object detection, Pruning, Convolutional neural network (CNN)

<img src="https://github.com/memgonzales/mirror-segmentation/blob/main/teaser.png?raw=True" alt="Teaser Figure" width = 800> 

## Running the Model

### Training
Run the following command to train the unpruned model:
```
python train.py
```

- The images should be saved in `<training_path>/image`.
- The ground-truth masks should be saved in `<training_path>/mask`.
- The ground-truth edge maps should be saved in `<training_path>/edge`.
- The training checkpoints will be saved in `<checkpoint_path>`. 
- `training_path` and `checkpoint_path` can be set in [`config.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/config.py).

To retrain the pruned model, follow the instructions in [`prune.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/prune.py).

### Prediction
Run the following command to perform prediction using the unpruned model:
```
python predict.py
```

Run the following command to perform prediction using the pruned model:
```
python prune.py
```

- The images should be saved in `<testing_path>/<dataset_name>/image`.
- The file path to the unpruned model weights should be `<weights_path>`.
- The file path to the pruned model weights should be `<pruned_weights_path>`.
- The predicted masks will be saved in `<result_path>/<dataset_name>`.
- `testing_path`, `dataset_name`, `weights_path`, `pruned_weights_path`, and `result_path` can be set in [`config.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/config.py).

### Evaluation
Run the following command to perform model evaluation:
```
python misc.py
```

- The predicted masks should be saved in `<result_path>/<dataset_name>`.
- The ground-truth masks should be saved in `<testing_path>/<dataset_name>/mask`.
- `result_path`, `testing_path`, and `dataset_name` can be set in [`config.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/config.py).


### Models & Weights
By default, [`train.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/train.py), [`predict.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/predict.py), and [`prune.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/prune.py) use the model defined in [`pmd.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/pmd.py), which employs an EfficientNetV2-Medium backbone and our proposed edge extraction and fusion module.

To explore the other feature extraction backbones that we considered in our experiments, refer to the models in [`models_experiments`](https://github.com/memgonzales/mirror-segmentation/tree/main/models_experiments) and the weights in this [Drive](https://drive.google.com/drive/folders/1Co488ztOdvY0208G2qPLBxOGIZxl3rjr?usp=sharing):

Model | Weights 
-- | --
**[Best]** [EfficientNetV2-Medium](https://github.com/memgonzales/mirror-segmentation/blob/main/pmd.py) | [Link](https://drive.google.com/file/d/1qq6SFD8Ve_4QQlSq7p0e1GVK7y0dhnP_/view?usp=sharing)
**[Best, Pruned]** [EfficentNetV2-Medium](https://github.com/memgonzales/mirror-segmentation/blob/main/prune.py) | [Link](https://drive.google.com/file/d/18zsqjK1aHVC4D8Ky530C--fwxdQylQ37/view?usp=sharing)
[ResNet-50](https://github.com/memgonzales/mirror-segmentation/blob/main/models_experiments/pmdvResNet.py) | [Link](https://drive.google.com/file/d/10_ZOeklWaGthscCe6mR-77l9E9RTwvuf/view?usp=sharing)
[ResNet-50 (+ PMD's original EDF module)](https://github.com/memgonzales/mirror-segmentation/blob/main/models_experiments/pmdvOrigResNet.py) | [Link](https://drive.google.com/file/d/10TRcRTvRLG6UuxryH7MJmZdTbAn_qiaH/view?usp=sharing)
[Xception-65](https://github.com/memgonzales/mirror-segmentation/blob/main/models_experiments/pmdvXception.py) | [Link](https://drive.google.com/file/d/1JgDjd4Au7CAy17ciWs2J6EgIHrPZ9o82/view?usp=sharing)
[VoVNet-39](https://github.com/memgonzales/mirror-segmentation/blob/main/models_experiments/pmdvVoVNet.py) | [Link](https://drive.google.com/file/d/14qRo1qyCZ32MDLxAN3pnMckdV2gKoGha/view?usp=sharing)
[MobileNetV3](https://github.com/memgonzales/mirror-segmentation/blob/main/models_experiments/pmdvMobileNet.py) | [Link](https://drive.google.com/file/d/1Z1aFh6HMMkm38RS3d_kcxjzvXs-OzXGT/view?usp=sharing)
[EfficientNet-Lite](https://github.com/memgonzales/mirror-segmentation/blob/main/models_experiments/pmdvEfficientNetLite.py) | [Link](https://drive.google.com/file/d/1K0GK4pOlOwKAlfwySHQhATTRJKQlhPbG/view?usp=sharing)
[EfficientNetEdge-Large](https://github.com/memgonzales/mirror-segmentation/blob/main/models_experiments/pmdvEfficientNetEdge.py) | [Link](https://drive.google.com/file/d/1W5H0sMuJOK0ylGUYhtY8m2rCkvumyzNi/view?usp=sharing)

*EDF stands for edge detection and fusion.* <br>

Note: With the exception of ResNet-50 (+ PMD's original EDF module), the models in the table above use our proposed edge extraction and fusion module.


## Dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7855135.svg)](https://doi.org/10.5281/zenodo.7855135)

**Our proposed dataset, DLSU-OMRS (De La Salle University &ndash; Outdoor Mirrors and Reflective Surfaces), can be downloaded from this [link](https://doi.org/10.5281/zenodo.7855135).** The images have their respective licenses, and the ground-truth masks are licensed under the [BSD 3-Clause "New" or "Revised" License](https://github.com/memgonzales/mirror-segmentation/blob/main/LICENSE). The use of this dataset is restricted to noncommercial purposes only.

The split PMD dataset, which we used for model training and evaluation, can be downloaded from this [link](https://drive.google.com/file/d/1_GrWcmRJndXd7wlB5tHqQjA3qx1J75xk/view). Our use of this dataset is under the BSD 3-Clause "New" or "Revised" License.

## Dependencies
The following Python libraries and modules (other than those that are part of the Python Standard Library) were used:

Library/Module | Description | License
-- | -- | --
[PyTorch](https://pytorch.org/) | Provides tensor computation with strong GPU acceleration and deep neural networks built on a tape-based autograd system | BSD 3-Clause License
[PyTorch Images Models](https://timm.fast.ai/) | Collection of state-of-the-art computer vision models, layers, and utilities | Apache License 2.0
[Neural Network Intelligence](https://nni.readthedocs.io/) | Provides tools for hyperparameter optimization, neural architecture search, model compression and feature engineering | MIT License
[Pillow](https://pillow.readthedocs.io/en/stable/) | Provides functions for opening, manipulating, and saving image files | Historical Permission Notice and Disclaimer
[scikit-image](https://scikit-image.org/) | Provides algorithms for image processing | BSD 3-Clause "New" or "Revised" License 
[PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf) | Python wrapper to dense (fully connected) conditional random fields with Gaussian edge potentials. | MIT License
[tqdm](https://tqdm.github.io/) | Allows the creation of progress bars by wrapping around any iterable	| Mozilla Public Licence (MPL) v. 2.0, MIT License
[NumPy](https://numpy.org/) | Provides a multidimensional array object, various derived objects, and an assortment of routines for fast operations on arrays | BSD 3-Clause "New" or "Revised" License 
[TensorBoardX](https://www.tensorflow.org/tensorboard) | Provides visualization and tooling needed for machine learning experimentation | MIT License

*The descriptions are taken from their respective websites.*

Note: Although PyDenseCRF can be installed via [`pip`](https://pypi.org/project/pydensecrf/) or its [official repository](https://github.com/lucasb-eyer/pydensecrf), we recommend Windows users to install it by running [`setup.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/pydensecrf/setup.py) inside the [`pydensecrf`](https://github.com/memgonzales/mirror-segmentation/tree/main/pydensecrf) directory of our repository to prevent potential issues with `Eigen.cpp` (refer to this [issue](https://github.com/lucasb-eyer/pydensecrf/issues/99) for additional details).

## Attributions
Attributions for reference source code are provided in the individual Python scripts and in the table below:

Reference | License
-- | --
[H. Mei, G. P. Ji, Z. Wei, X. Yang, X. Wei, and D. P. Fang (2021). "Camouflaged object segmentation with distraction mining," in *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. Nashville, TN, USA: IEEE Computer Society, June 2021, pp. 8768–8877.](https://github.com/Mhaiyang/CVPR2021_PFNet) | BSD 3-Clause "New" or "Revised" License
[J. Wei, S. Wang, and Q. Huang, "F³net: Fusion, feedback and focus for salient object detection," *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 07, pp. 12321–12328, Apr. 2020.](https://github.com/weijun88/F3Net) | MIT License
[J. Lin, G. Wang, and R. H. Lau, "Progressive mirror detection,” in *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. Los Alamitos, CA, USA: IEEE Computer Society, June 2020, pp. 3694–3702.](https://jiaying.link/cvpr2020-pgd/) | BSD 3-Clause "New" or "Revised" License


## Authors

-   <b>Mark Edward M. Gonzales</b> <br/>
    mark_gonzales@dlsu.edu.ph <br/> 
    
-   <b>Lorene C. Uy</b> <br/>
    lorene_c_uy@dlsu.edu.ph <br/>
    
-   <b>Dr. Joel P. Ilao</b> <br/>
    joel.ilao@dlsu.edu.ph <br/>    

This is the major course output in a computer vision class for master's students under Dr. Joel P. Ilao of the Department of Computer Technology, De La Salle University. The task is to create an eight-week small-scale project that applies computer vision-based techniques to present a solution to an identified research problem.

[badge-python]: https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=white
[badge-numpy]: https://img.shields.io/badge/Numpy-777BB4?style=flat&logo=numpy&logoColor=white
[badge-github-actions]: https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white
