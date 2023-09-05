# Denoising diffusion models for graph generation

Warning 1: when running the code, you might encounter an `AttributeError cls._old_init`. This is a non deterministic
error due to a bad interaction between `pytorch_lightning` and `torch_geometric`. Just run the code again until it works
(it might happen up to 5 times in a row)

Warning 2: The code has been updated since experiments were run for the paper. If you don't manage to reproduce the 
paper results, please write to us so that we can investigate the issue.

Warning 3: the conditional generation experiments were implemented with an legacy version of the code. They are not yet available in the public version.

## Environment installation
  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit: `conda create -c conda-forge -n my-rdkit-env rdkit`
  - Install graph-tool (https://graph-tool.skewed.de/)
  - Install pytorch 1.10 or 1.11 (https://pytorch.org/)
  - Install pytorch-geometric. Your version should match the pytorch version that is installed (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  - Install other packages using the requirement file: `pip install -r requirements.txt`
  - Install mini-moses: `pip install git+https://github.com/igor-krawczuk/mini-moses@main`

## Download the data

  - QM9 and Guacamol should download by themselves when you run the code.
  - For the community, SBM and planar datasets, data can be found at https://github.com/KarolisMart/SPECTRE/tree/main/data
  - Moses dataset can be found at https://github.com/molecularsets/moses/tree/master/data
  




## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - To run the continuous model: `python3 main.py model=continuous`
  - To run the discrete model: `python3 main.py`
  - You can specify the dataset with `python3 main.py dataset=guacamol`. Look at `configs/dataset` for the list
of datasets that are currently available
    
    
## Cite the paper

```
@article{vignac2022digress,
  title={DiGress: Discrete Denoising diffusion for graph generation},
  author={Vignac, Clement and Krawczuk, Igor and Siraudin, Antoine and Wang, Bohan and Cevher, Volkan and Frossard, Pascal},
  journal={arXiv preprint arXiv:2209.14734},
  year={2022}
}
```


## Current problems 
- Validity of molecules is wrong, most of them are plain wrong (probably not checking if a bond change the n of allowed bonds of an atom) -> Fixed
- No 3D
- Really bad way to add bond 
- No frag dataset -> IN PROGRESS, ZINC preprocessing 
- One hot embedding -> Change to nn.Embedding
- Clean a lil the API and allows to load the model more easily

## TODO
- Understanding this code
- Replacing the codebook of fragments with edge prediction and including 3D (see papers)
- Rework the dataset class for the frag dataset

## Models

| Model  | Config  | Checkpoint  | Regr Conditional  | Wandb \
|---|---|---|---|---|
| Toy model 1 | /checkpoints/toy_models/discrete_toy_model_1/config.yaml | /checkpoints/toy_models/discrete_toy_model_1/discrete_epoch=1234.ckpt | No   | [Toy](https://wandb.ai/fntwin/SAFE_SPACE/runs/4wanrsvm?workspace=user-fntwin) |
|   |   |   |   | |
|   |   |   |   | |