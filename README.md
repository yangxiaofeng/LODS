# Learn to Optimize Denoising Scores for 3D Generation: A Unified and Improved Diffusion Prior on NeRF and 3D Gaussian Splatting
### [Project Page](https://yangxiaofeng.github.io/demo_diffusion_prior/) | [Arxiv Paper]()




We propose a unified framework aimed at enhancing the diffusion priors for 3D generation tasks. Despite the critical importance of these tasks, existing methodologies often struggle to generate high-caliber results. We begin by examining the inherent limitations in previous diffusion priors. We identify a divergence between the diffusion priors and the training procedures of diffusion models that substantially impairs the quality of 3D generation. To address this issue, we propose a novel, unified framework that iteratively optimizes both the 3D model and the diffusion prior. Leveraging the different learnable parameters of the diffusion prior, our approach offers multiple configurations, affording various trade-offs between performance and implementation complexity. Notably, our experimental results demonstrate that our method markedly surpasses existing techniques, establishing new state-of-the-art in the realm of text-to-3D generation. Furthermore, our approach exhibits impressive performance on both NeRF and the newly introduced 3D Gaussian Splatting backbones. Additionally, our framework yields insightful contributions to the understanding of recent score distillation methods, such as the VSD and DDS loss.
## Updates
- 07/12/2023: Code Released.




## Installation
Our codes are based on the implementations of [ThreeStudio](https://github.com/threestudio-project/threestudio) and [GaussianDreamer](https://github.com/hustvl/GaussianDreamer).
Please follow the instructions from two links to install the project.

## Quickstart
### 2D Playground
```
python 2dplayground_lora.py
python 2dplayground_embedding.py
```
You should get results similar to these:

|      LoRA       |  Embedding |
|:-------------------------:|:-------------------------:|
| ![](images/lora_2d.png)  |  ![](images/embedding_2d.png)|





### Run LODS Embedding + 3D Gaussian Splatting
```
python launch.py --config configs/lods-gs-embedding.yaml --train --gpu 0 system.prompt_processor.prompt="a DSLR image of a hamburger" 
```
### Run LODS LoRA + Instant-NGP
```
python launch.py --config configs/lods-ngp-lora.yaml --train --gpu 0 system.prompt_processor.prompt="a DSLR image of a hamburger"
```
### Run LODS Embedding + Instant-NGP
```
python launch.py --config configs/lods-ngp-embedding.yaml --train --gpu 0 system.prompt_processor.prompt="a DSLR image of a hamburger"
```


