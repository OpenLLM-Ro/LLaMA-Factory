# Training of RoLLMs

Official code used for training Romanian LLMs as proposed in [Masala et al. 2024](https://arxiv.org/abs/2406.18266). This repo is a fork of the popular [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repo used for training and finetuning LLMs. On top of the existing framework we added a suite of Romanian datasets:

- [ro_sft_alpaca](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_alpaca)
- [ro_sft_alpaca_gpt4](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_alpaca_gpt4)
- [ro_sft_dolly](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_dolly)
- [ro_sft_selfinstruct_gpt4](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_selfinstruct_gpt4)
- [ro_sft_norobts](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_norobots)
- [ro_sft_orca](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_orca)
- [ro_sft_camel](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_camel)


![# LLaMA Factory](assets/logo.png)



- [Requirement](#requirement)
- [Getting Started](#getting-started)


## Requirement

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.9     | 3.10      |
| torch        | 2.0.0   | 2.6.0     |
| torchvision  | 0.15.0  | 0.21.0    |
| transformers | 4.49.0  | 4.50.0    |
| datasets     | 2.16.0  | 3.2.0     |
| accelerate   | 0.34.0  | 1.2.1     |
| peft         | 0.14.0  | 0.15.1    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.16.4    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.8.2     |
| flash-attn   | 2.5.6   | 2.7.2     |

### Hardware Requirement

\* *estimated*

| Method                          | Bits |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | ---- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |  32  | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              |  16  |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |   8  |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |   4  |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |   2  |   4GB |   8GB |  16GB |   24GB | `x/4`GB |

## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Extra dependencies available: torch, torch-npu, metrics, deepspeed, liger-kernel, bitsandbytes, hqq, eetq, gptq, awq, aqlm, vllm, galore, apollo, badam, adam-mini, qwen, minicpm_v, modelscope, openmind, swanlab, quality

> [!TIP]
> Use `pip install --no-deps -e .` to resolve package conflicts.

### Data Preparation

Please refer to [data/README.md](data/README.md) for checking the details about the format of dataset files. You can either use datasets on HuggingFace / ModelScope / Modelers hub or load the dataset in local disk.

> [!NOTE]
> Please update `data/dataset_info.json` to use your custom dataset.

### Quickstart

Use the following 3 commands to run LoRA **fine-tuning**, **inference** and **merging** of the Llama3-8B-Instruct model, respectively.

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

See [examples/README.md](examples/README.md) for advanced usage (including distributed training).

> [!TIP]
> Use `llamafactory-cli help` to show help information.

### Use W&B Logger

To use [Weights & Biases](https://wandb.ai) for logging experimental results, you need to add the following arguments to yaml files.

```yaml
report_to: wandb
run_name: test_run # optional
```

Set `WANDB_API_KEY` to [your key](https://wandb.ai/authorize) when launching training tasks to log in with your W&B account.

### Use SwanLab Logger

To use [SwanLab](https://github.com/SwanHubX/SwanLab) for logging experimental results, you need to add the following arguments to yaml files.

```yaml
use_swanlab: true
swanlab_run_name: test_run # optional
```

When launching training tasks, you can log in to SwanLab in three ways:

1. Add `swanlab_api_key=<your_api_key>` to the yaml file, and set it to your [API key](https://swanlab.cn/settings).
2. Set the environment variable `SWANLAB_API_KEY` to your [API key](https://swanlab.cn/settings).
3. Use the `swanlab login` command to complete the login.


## Citation


```bibtex
@misc{masala2024vorbecstiromanecsterecipetrain,
      title={"Vorbe\c{s}ti Rom\^ane\c{s}te?" A Recipe to Train Powerful Romanian LLMs with English Instructions}, 
      author={Mihai Masala and Denis C. Ilie-Ablachim and Alexandru Dima and Dragos Corlatescu and Miruna Zavelca and Ovio Olaru and Simina Terian and Andrei Terian and Marius Leordeanu and Horia Velicu and Marius Popescu and Mihai Dascalu and Traian Rebedea},
      year={2024},
      eprint={2406.18266},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18266}, 
}
```

### Acknowledgement
This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank them for their wonderful work.


