# Apply Alpaca method to GPT-J-6B Text Summarization Downstream

This codebase is adapted from [standford alpaca](https://github.com/tatsu-lab/stanford_alpaca) repo, you can refer to [here](https://github.com/tatsu-lab/stanford_alpaca/blob/main/README.md) for their original README, here we only put this task's README.

## Overview

### dataset
We use [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset hosted by HuggingFace, version 3.0.0.

### base model
We use [EleutherAI's pretrained gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) as our base model.

### hyper-parameters
We refer to [this guideline](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md) to tune hyper-parameters.

## Step-by-step
### pull the docker
```shell
docker pull yaomatrix/alpaca:2
```

### clone the code
```shell
cd <your_working_dir>
git clone https://github.com/intel-sandbox/text_summarization_gpt-j-6b.git
```

### run the NV GPU docker
```shell
docker run --gpus all -it --rm --privileged  --ulimit memlock=-1 --ulimit stack=67108864  --network host --pid=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE  --shm-size=1g -v <your_repo_path>:/workspace yaomatrix/alpaca:2 /bin/bash
```

> **The following steps are all done in docker container**

### generate dataset

> There is a generated dataset in ./data folder of the repo, you can use it directly or generate by your own

```shell
python ./cnndailymail_data_prep.py
```

### train
> `8 x A100-80G` single node

```shell
bash ./train.sh
```

### evaluation

> `8 x A100-80G` single node

```shell
bash ./eval.sh
```
