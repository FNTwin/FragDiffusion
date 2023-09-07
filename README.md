# Denoising diffusion models for graph generation

Warning 1: when running the code, you might encounter an `AttributeError cls._old_init`. This is a non deterministic
error due to a bad interaction between `pytorch_lightning` and `torch_geometric`. Just run the code again until it works
(it might happen up to 5 times in a row)

Warning 2: The code has been updated since experiments were run for the paper. If you don't manage to reproduce the 
paper results, please write to us so that we can investigate the issue.

Warning 3: the conditional generation experiments were implemented with an legacy version of the code. They are not yet available in the public version.

Warning 4: (Cristian) MultiGPU for the fragdiff repo is under development

## Environment installation

Install it with _mamba_ 🐍

```bash
# Install the deps
mamba env create -n  fragdiff -f env.yml

# Activate the environment
mamba activate fragdiff

# Install the library in dev mode
pip install -e .
```

## Run the fragdiff code 

`python3 main.py dataset=frag`

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
| Toy model 1 | /checkpoints/toy_models/discrete_toy_model_1/config.yaml | /checkpoints/toy_models/discrete_toy_model_1/discrete_epoch=1234.ckpt | No   | [Toy 1](https://wandb.ai/fntwin/SAFE_SPACE/runs/4wanrsvm?workspace=user-fntwin) |
| Toy model 2 |   | /home/cristian_valencediscovery_com/dev/FragDiffusion/dgd/expts/outputs/2023-09-05/14-41-22/checkpoints/run_1000dsteps_128batch/epoch=759.ckpt  | dgd/expts/outputs/2023-09-05/14-41-22/.hydra/config.yaml  | [Toy 2](https://wandb.ai/fntwin/SAFE_SPACE/runs/ulpux8qn?workspace=user-fntwin) |
| Paper Feat ALL  |   |   |   | [All] () |
| Paper Feat None   |   |   |   | [None] ()|
| Paper Feat Cycles   |   |   |   | [Cycles] ()|