# Designing a Lightweight Edge-Guided CNN for Segmenting Mirrors and Reflective Surfaces
![badge][badge-python]
![badge][badge-numpy]
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)  <br>
[![Actions Status](https://github.com/bioinfodlsu/phage-host-prediction/workflows/Check%20for%20syntax%20errors/badge.svg)](https://github.com/bioinfodlsu/phage-host-prediction/actions)
![badge][badge-github-actions]

## Table of Contents
- [Description](https://github.com/memgonzales/mirror-segmentation#description)
- [Running the Model](https://github.com/memgonzales/mirror-segmentation#running-the-model)
  - [Training](https://github.com/memgonzales/mirror-segmentation#training)
  - [Prediction](https://github.com/memgonzales/mirror-segmentation#prediction)
  - [Evaluation](https://github.com/memgonzales/mirror-segmentation#evaluation)
  - [Models & Weights](https://github.com/memgonzales/mirror-segmentation#models--weights)
- [Dataset](https://github.com/memgonzales/mirror-segmentation#dataset)
- [Dependencies](https://github.com/memgonzales/mirror-segmentation#dependencies)
- [Authors](https://github.com/memgonzales/mirror-segmentation#authors)

## Description
**ABSTRACT:** The detection of mirrors is a challenging task due to their lack of a distinguishing appearance and the visual similarity of reflections with their surroundings. While existing systems have achieved some success in mirror segmentation, the design of lightweight models remains unexplored, and datasets are mostly limited to clear mirrors in indoor scenes. In this paper, we propose a new dataset consisting of 454 images of outdoor mirrors and reflective surfaces. We also present a lightweight edge-guided convolutional neural network based on PMDNet. Our model uses EfficientNetV2-Medium as the backbone, and employs parallel convolutional layers and a lightweight convolutional block attention module to capture both low-level and high-level features for edge extraction. It registered $F_\beta$ scores of 0.8483, 0.8117, and 0.8388 on MSD, PMD, and our proposed dataset, respectively. Applying filter pruning via geometric median resulted in $F_\beta$ scores of 0.8498, 0.7902, and 0.8456, respectively, performing competitively with the state-of-the-art PMDNet but with 78.20&times; fewer FLOPS and 238.16&times; fewer parameters.

**Index Terms:** Mirror segmentation, Object detection, Pruning, Convolutional neural network (CNN)

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
- `training_path` can be set in [`config.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/config.py).

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
- `testing_path`, `dataset_name`, `weights_path`, and `pruned_weights_path` can be set in [`config.py`](https://github.com/memgonzales/mirror-segmentation/blob/main/config.py).

### Evaluation
- The ground-truth masks should be saved in `<testing_path>/<dataset_name>/mask`.


### Models & Weights
By default, `train.py`, `predict.py`, and `prune.py` refer to the model defined in `pmd.py`. 

## Dataset
**Our proposed dataset, DLSU-OMRS (De La Salle University &ndash; Outdoor Mirrors and Reflective Surfaces), can be downloaded from this [link](https://drive.google.com/drive/folders/1UekoWvJQQr9UoTIFoQuyX3Y7X80_zkW_?usp=sharing).** The images have their respective licenses, and the ground-truth masks are licensed under the [BSD 3-Clause "New" or "Revised" License](https://github.com/memgonzales/mirror-segmentation/blob/main/LICENSE). The use of this dataset is restricted to noncommercial purposes only.

The split PMD dataset, which we used for model training and evaluation, can be downloaded from this [link](https://drive.google.com/file/d/1_GrWcmRJndXd7wlB5tHqQjA3qx1J75xk/view). Our use of this dataset is under the BSD 3-Clause "New" or "Revised" License.

## Dependencies

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
