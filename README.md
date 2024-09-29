# CHGGen

This is a repository for `CHGGen`: an SE(3)--equivariant diffusion model built upon `e3nn` and `NequIP` frameworks. 

* For local environment sampling, inpainting-based conditional generation is used (see the related work on [ICML 2024 AI for Science Workshop](https://openreview.net/forum?id=T1mIt5exUF&noteId=T1mIt5exUF))
* For general amorphous or small crystal structure generation, _de novo_ (unconditional) generation is used (with reversed SDE)

## Installation

The model be run on a GPU machine. For example, one can install the relevant packages using the following commands
```
conda create -n enhan_gen python=3.9
conda activate enhan_gen

pip3 install torch torchvision torchaudio 

pip3 install pytorch-lightning torch_geometric 

pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

pip install matplotlib ase pymatgen==2023.9.10 p_tqdm cvxpy

pip install e3nn==0.5.1
```


## Acknowledgements

If you find this package useful, please consider the citation:
```
@inproceedings{crys_inpainting_2024,
title={Inpainting crystal structure generations with score-based denoising},
author={Xinzhe Dai, Peichen Zhong, Bowen Deng, Yifan Chen, Gerbrand Ceder},
booktitle={ICML 2024 AI for Science Workshop},
year={2024},
url={https://openreview.net/forum?id=T1mIt5exUF}
}
```

The development of `enhan_gen` and `CHGGen` used/referenced the implementation of the following packages. Please consider citing the relevant works:
* [CDVAE](https://arxiv.org/abs/2110.06197)
* [Graphite](https://arxiv.org/abs/2212.02421)
* [E3NN](https://arxiv.org/abs/2207.09453)
* [NequIP](https://www.nature.com/articles/s41467-022-29939-5)
