# CHGGen

This is a repository for `CHGGen`: Crystal Host-Guided Generation built upon `e3nn` and `NequIP` frameworks. 

## Installation

The model be run on a GPU machine. For example, one can install the relevant packages using the following commands
```
conda create -n chggen
conda activate chggen

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121 

pip3 install pytorch-lightning torch_geometric 

pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

pip install matplotlib ase pymatgen==2023.9.10 p_tqdm cvxpy scikit-learn

pip install e3nn==0.5.1

pip install -e .
```


## Acknowledgements

If you find this package useful, please consider the citation:
```

@article{chggen,
title={{Practical approaches for crystal structure predictions with inpainting generation and universal interatomic potentials}},
author={Zhong, Peichen and Dai, Xinzhe and Deng, Bowen and Ceder, Gerbrand and Persson, Kristin},
journal = {arXiv preprint arXiv:2504.16893},
URL = {https://arxiv.org/abs/2504.16893},
author+an =	 {1=highlight,corresponding;5=corresponding}
}


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
