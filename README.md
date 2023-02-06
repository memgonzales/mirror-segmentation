# Designing a Lightweight Edge-Guided CNN for Segmenting Mirrors and Reflective Surfaces

## Table of Contents
- [Description](https://github.com/memgonzales/mirror-segmentation#description)
- [Dataset](https://github.com/memgonzales/mirror-segmentation#dataset)
- Running the Model
  - Training
  - Testing
  - Models & Weights
- Dependencies
- [Authors](https://github.com/memgonzales/mirror-segmentation#authors)

## Description
**ABSTRACT:** The detection of mirrors is a challenging task due to their lack of a distinguishing appearance and the visual similarity of reflections with their surroundings. While existing systems have achieved some success in mirror segmentation, the design of lightweight models remains unexplored, and datasets are mostly limited to clear mirrors in indoor scenes. In this paper, we propose a new dataset consisting of 454 images of outdoor mirrors and reflective surfaces. We also present a lightweight edge-guided convolutional neural network based on PMDNet. Our model uses EfficientNetV2-Medium as the backbone, and employs parallel convolutional layers and a lightweight convolutional block attention module to capture both low-level and high-level features for edge extraction. It registered $F_\beta$ scores of 0.8483, 0.8117, and 0.8388 on MSD, PMD, and our proposed dataset, respectively. Applying filter pruning via geometric median resulted in $F_\beta$ scores of 0.8498, 0.7902, and 0.8456, respectively, performing competitively with the state-of-the-art PMDNet but with 78.20&times; fewer FLOPS and 238.16&times; fewer parameters.

**Index Terms:** Mirror segmentation, Object detection, Pruning, Convolutional neural network (CNN)

<img src="https://github.com/memgonzales/mirror-segmentation/blob/main/teaser.png?raw=True" alt="Teaser Figure" width = 800> 

*↑ Return to [Table of Contents](https://github.com/memgonzales/mirror-segmentation#table-of-contents).*

## Running the Model

### Training

*↑ Return to [Table of Contents](https://github.com/memgonzales/mirror-segmentation#table-of-contents).*

### Prediction

*↑ Return to [Table of Contents](https://github.com/memgonzales/mirror-segmentation#table-of-contents).*

### Models & Weights

*↑ Return to [Table of Contents](https://github.com/memgonzales/mirror-segmentation#table-of-contents).*

## Dataset
**Our proposed dataset, DLSU-OMRS (De La Salle University &ndash; Outdoor Mirrors and Reflective Surfaces), can be downloaded from this [link](https://drive.google.com/drive/folders/1UekoWvJQQr9UoTIFoQuyX3Y7X80_zkW_?usp=sharing).** The images have their respective licenses, and the ground-truth masks are licensed under the [BSD 3-Clause "New" or "Revised" License](https://github.com/memgonzales/mirror-segmentation/blob/main/LICENSE). The use of this dataset is restricted to noncommercial purposes only.

The split PMD dataset, which we used for model training and evaluation, can be downloaded from this [link](https://drive.google.com/file/d/1_GrWcmRJndXd7wlB5tHqQjA3qx1J75xk/view). Our use of this dataset is under the BSD 3-Clause "New" or "Revised" License.

*↑ Return to [Table of Contents](https://github.com/memgonzales/mirror-segmentation#table-of-contents).*

## Authors

-   <b>Mark Edward M. Gonzales</b> <br/>
    mark_gonzales@dlsu.edu.ph <br/>
-   <b>Lorene C. Uy</b> <br/>
    lorene_c_uy@dlsu.edu.ph <br/>
-   <b>Dr. Joel P. Ilao</b> <br/>
    joel.ilao@dlsu.edu.ph <br/>    

This is the major course output in a computer vision class for master's students under Dr. Joel P. Ilao of the Department of Computer Technology, De La Salle University. The task is to create an eight-week small-scale project that applies computer vision-based techniques to present a solution to an identified research problem.
