# LiDAL: Inter-frame Uncertainty Based Active Learning for 3D LiDAR Semantic Segmentation

![Framework Fig](docs/LiDAL_ECCV2022.png)

Created by Zeyu HU

## Introduction
This work is based on our paper 
[LiDAL: Inter-frame Uncertainty Based Active Learning for 3D LiDAR Semantic Segmentation](https://arxiv.org/abs/2211.05997),
which appears at the European Conference on Computer Vision (ECCV) 2022. 

We propose LiDAL, a novel active learning method for 3D LiDAR semantic segmentation by exploiting inter-frame uncertainty among LiDAR frames. Our core idea is that a well-trained model should generate robust results irrespective of viewpoints for scene scanning and thus the inconsistencies in model predictions across frames provide a very reliable measure of uncertainty for active sample selection. To implement this uncertainty measure, we introduce new inter-frame divergence and entropy formulations, which serve as the metrics for active selection. Moreover, we demonstrate additional performance gains by predicting and incorporating pseudo-labels, which are also selected using the proposed inter-frame uncertainty measure. Experimental results validate the effectiveness of LiDAL: we achieve 95% of the performance of fully supervised learning with less than 5% of annotations on the SemanticKITTI and nuScenes datasets, outperforming state-of-the-art active learning methods.

## Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{hu2022lidal,
    title={LiDAL: Inter-frame Uncertainty Based Active Learning for 3D LiDAR Semantic Segmentation},
    author={Hu, Zeyu and Bai, Xuyang and Zhang, Runze and Wang, Xin and Sun, Guangyuan and Fu, Hongbo and Tai, Chiew-Lan},
    booktitle={European Conference on Computer Vision},
    pages={248--265},
    year={2022},
    organization={Springer}
    }

## Installation
* For linux-64 platform with conda, the enviroment can be created by [requirements.txt](docs/requirements.txt): 
      
      conda create --name <env> --file <this file>

* Our code is based on <a href="https://pytorch.org/">Pytorch</a>. Please make sure <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html">CUDA</a>  and <a href="https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html">cuDNN</a> are installed. One configuration has been tested: 
     - Python 3.9.6
     - Pytorch 1.9.0
     - torchvision 0.10.0
     - CUDA 10.2
     - cudatoolkit 10.2.89
     - cuDNN 7.6.5

* VMNet depends on the <a href="https://github.com/mit-han-lab/torchsparse">torchsparse</a> library. Please follow its installation instructions. One configuration has been tested: 

     - torchsparse 1.4.0
 
## Data Preparation
* Please refer to http://www.semantic-kitti.org/ and https://www.nuscenes.org/ to get access to the SemanticKITTI and nuScenes dataset. This code is based on their official file orginizations. Please put the files in directories 'Semantic_kitti/' and 'nuScenes/' accordingly.

* Prepare sub-scene regions. [k-means-constrained](https://github.com/joshlk/k-means-constrained)
      
      python -m dataset.prepare_supervoxel_kmeans_sk
      python -m dataset.prepare_supervoxel_kmeans_nu

* Prepare kdtree.
      
      python -m dataset.prepare_kdtree_sk
      python -m dataset.prepare_kdtree_nu

* [ReDAL](https://github.com/tsunghan-wu/ReDAL)

    To implement the ReDAL method in this code, the sub-scene regions are divided by the VCCS algorithm.
    
    - Dependency
      - [Point CLoud Library (PCL)](https://pointclouds.org/)
      - [Boost C++ Library](https://www.boost.org/)
      - [CMake](https://cmake.org/)
      - [cnpy](https://github.com/rogersce/cnpy)
      
    - Build the project in pcl_related/ via CMake

    Then run:

      python -m dataset.prepare_supervoxel_VCCS_sk
      python -m dataset.prepare_supervoxel_VCCS_nu  

    Also, it requires the calculation of surface variation.
      
      python -m dataset.ReDAL.gen_surface_variation_sk
      python -m dataset.ReDAL.gen_surface_variation_nu


## Run
* Fully supervised learning.
    
      CUDA_VISIBLE_DEVICES=X python train.py --dataset_name [SK, NU] --model_name [SPVCNN, Mink] --label_unit fr --metric_name full --r_id -1
      
* Zero round initialization. 

      CUDA_VISIBLE_DEVICES=X python train.py --dataset_name [SK, NU] --model_name [SPVCNN, Mink] --label_unit fr --metric_name None --r_id 0

    - For bechmarking purposes, the initial models used in our experiments can be found at:
      - <a href="https://drive.google.com/file/d/1LUSTJ-YgK5qrTa9uucOkf2tkywJFjrUR/view?usp=sharing">SK_SPVCNN_0r</a>
      - <a href="https://drive.google.com/file/d/1s6QLQg15Mn3XcWdhKYiIn_umGRsrPKGV/view?usp=sharing">SK_Mink_0r</a>
      - <a href="https://drive.google.com/file/d/1o9vlKa-30ft3FbgAFTd6VwxFC_Y9UEU_/view?usp=sharing">NU_SPVCNN_0r</a>
      - <a href="https://drive.google.com/file/d/1W9IXjtZnHFn6D_EkViNPqmdXXNHln-y-/view?usp=sharing">NU_Mink_0r</a> 

* K round active learning.
    - Supported active method flags: 
      - Frame-based
        - Random Selection (RAND)
        - Segment-entropy (SEGENT)
        - Softmax Margin (MAR)
        - Softmax Confidence (CONF)
        - Softmax Entropy (ENT)
        - Core-set Selection (CSET)
      - Region-based
        - Random Selection (RAND)
        - ReDAL
        - LiDAL (ours)

    1. Probability inference of current trained model (K - 1 round). 

      CUDA_VISIBLE_DEVICES=X python -m score.prob_inference --dataset_name [SK, NU] --model_name [SPVCNN, Mink] --label_unit [fr, sv] --metric_name [...] --r_id (K-1)
    
    2. Uncertainty scoring and sample selection. Examples below:

      python -m score.frame_level.core_set --dataset_name [SK, NU] --model_name [SPVCNN, Mink] --r_id K
      python -m score.sv_level.LiDAL --dataset_name [SK, NU] --model_name [SPVCNN, Mink] --r_id K
      
    3. Training of K round.

      CUDA_VISIBLE_DEVICES=X python train.py --dataset_name [SK, NU] --model_name [SPVCNN, Mink] --label_unit [fr, sv] --metric_name [...] --r_id K

    4. Validation of K round.

      CUDA_VISIBLE_DEVICES=X python evaluate.py --dataset_name [SK, NU] --model_name [SPVCNN, Mink] --label_unit [fr, sv] --metric_name [...] --r_id K

