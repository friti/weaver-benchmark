# weaver-benchmark

[`Weaver`](https://github.com/hqucms/weaver-core) configurations for ML benchmark tasks

## Set up the environment 
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.1.2-0-Linux-x86_64.sh
bash Miniconda3-py39_24.1.2-0-Linux-x86_64.sh
# Follow the instructions to finish the installation

# Make sure to choose `yes` for the following one to let the installer initialize Miniconda3
# > Do you wish the installer to initialize Miniconda3
# > by running conda init? [yes|no]

# disable auto activation of the base environment
conda config --set auto_activate_base false

# create conda anvironment weaver
conda create -n weaver python=3.11
conda activate weaver

pip3 install numpy 
pip3 install scikit-learn scipy matplotlib tqdm
pip3 install PyYAML beautifulsoup4 lz4 xxhash tables 
pip3 install vector tensorboard
pip3 install uproot awkward awkward0
pip3 install onnx onnxruntime-gpu onnxruntime
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 cache purge

conda install -c conda-forge root

```

## Prepare for running on condor
```
git clone git@github.com:friti/weaver-core.git -b domain_adaptation
git clone git@github.com:friti/weaver-benchmark.git -b softditau
ln -s <path-to-weaver-benchmark>/tau_tagging <path-to-weaver-core>/weaver-core/weaver/
ln -s <path-to-weaver-core>/weaver-core/ <path-to-weaver-benchmark>/weaver-benchmark/

cd weaver-benchmark/weaver-core/weaver
mkdir output
cd ../../condor
mkdir jobs_log

# check the paths in run.sh (both beginning and the end of the script)
condor_submit submit.sub
```

## Pre-processing for weights

Weaver can be run locally on CPUs only to produced a new `data-config` file containing weights for the chosen binning and selection. Everytime you change the input datasets or the selection or the binning defintion or the classes, these files need to be reporduced i.e.:
* Delete them from both git and local directory, since the hash-key of the file will be generated differently from execution to execution.
* Rerun the training aborting it after the production of the data-config file after the pre-processing of the inputs.
* Example of commands:
  ```sh
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_ext.py --data-config data/ak4_points_pf_sv_mass_decorr.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_notau.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_ext.py --data-config data/ak4_points_pf_sv_mass_decorr_tau.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_tau.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_ext.py --data-config data/ak4_points_pf_sv_mass_decorr_taumuel.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_taumuel.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_class_reg.py --data-config data/ak4_points_pf_sv_mass_decorr_tau_class_reg.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_tau_classreg.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK4Skimmed/tree_*root'  --network-config networks/particle_net_ak4_pf_sv_class_reg.py --data-config data/ak4_points_pf_sv_mass_decorr_taumuel_class_reg.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak4_taumuel_classreg.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_ne_ak8_pf_sv_ext.py --data-config data/ak8_points_pf_sv_mass_decorr.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_notau.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_net_ak8_pf_sv_ext.py --data-config data/ak8_points_pf_sv_mass_decorr_tau.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_tau.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_net_ak8_pf_sv_class_reg.py --data-config data/ak8_points_pf_sv_mass_decorr_tau_class_reg.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_tau_classreg.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_net_ak8_pf_sv_ext.py --data-config data/ak8_points_pf_sv_mass_decorr_taumuel.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_taumuel.log --num-workers 1
  python3 ../weaver-core/weaver/train.py  --data-train '/eos/cms/store/group/phys_exotica/monojet/rgerosa/ParticleNetUL/NtupleTrainingAK8Skimmed/tree_*root'  --network-config networks/particle_net_ak8_pf_sv_class_reg.py --data-config data/ak8_points_pf_sv_mass_decorr_taumuel_class_reg.yaml --model-prefix /tmp/rgerosa/ --gpus '' --batch-size 100 --log /tmp/rgerosa/weight_ak8_taumuel_classreg.log --num-workers 1
```

