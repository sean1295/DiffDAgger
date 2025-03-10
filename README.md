# DiffDAgger

Please note that the code itself it not super optimized for performance. Feel free to make changes for improvement. If you have any questions, please contact dcs3zc@virginia.edu to reach out to me.

## Installation ☑️

### Installing using conda🗜

You can install the vitual conda environment using the following command:
```
conda env create -f environment.yml
```
## Simulation

### Installing Dependencies

For simulation runs, run the following to install a forked version of Maniskill3. For details regarding requirements for Maniskill3, including the usage of Vulkan driver, visit https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html. 
```
cd mani_skill
pip install -e .
cd ../
```

To run Diff-DAgger in simulated push-T task:
```
python diffdagger/sim_train.py
```


## Real World
### Dataset Format
To run the example code to see the dataset format:
```
python tests/example_dataset.py
```

### Data Collection
To see the environment format and collect data with a simulated expert, run:
```
python tests/example_data_collection.py
```

### Training

In this project, hugging face accelerate [link](https://huggingface.co/docs/accelerate/en/index) is used to train with distributed process. 


#### 1. Module requirement

The current compatible version for CUDA, cuDNN, and gcc to run the accelerate pipeline are:

```
module load cudnn/8.9.7
module load cuda/11.8.0
module load gcc/13.3.0
```

#### 2. Distributed Training
Run the following script to train the diffusion policy AFTER collecting the data and specifying the dataset_dir inside the yml file used in train.py (diffdagger/config/test/bread_toast.yml in this case).

```
accelerate launch --main_process_port 6006 diffdagger/train.py
```

This script will save the policy after every 5k training steps after 50k initial steps.

### Evaluation
AFTER you specify the path for where the policy is saved in the yml file,

```
(ex) load_file: ${dataset_dir}/eef_pose_v_prediction_r3m_unet_diffstep64_100k.pth
```
run the example code to see the environment class and how the policy generates the actions:
```
python tests/example_rollout.py
```
