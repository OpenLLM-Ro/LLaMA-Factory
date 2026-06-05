# Training of RoLLMs

Official code used for training Romanian LLMs as proposed in [Masala et al. 2024](https://arxiv.org/abs/2406.18266). This repo is a fork of the popular [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repo used for training and finetuning LLMs. On top of the existing framework we added a suite of Romanian datasets:

- [ro_sft_alpaca](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_alpaca)
- [ro_sft_alpaca_gpt4](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_alpaca_gpt4)
- [ro_sft_dolly](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_dolly)
- [ro_sft_selfinstruct_gpt4](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_selfinstruct_gpt4)
- [ro_sft_norobts](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_norobots)
- [ro_sft_orca](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_orca)
- [ro_sft_camel](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_camel)

> [!IMPORTANT]
> **Reproducing the paper (text-only).** This `main` branch has since been updated to a newer LLaMA-Factory base and **extended with vision-language (VLM) training** (LLaVA-NeXT, Gemma3, Qwen3-VL) alongside the original text-only work. For the **exact code used in [Masala et al. 2024]**, check out the frozen tag:
> ```bash
> git checkout v1.0-text
> ```

### Vision-language datasets

This repo also adds a suite of Romanian **vision-language** SFT datasets:

- [ro_sft_cosyn](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_cosyn)
- [ro_sft_finepdfs](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_finepdfs)
- [ro_sft_flickr30k_cap](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_flickr30k_cap)
- [ro_sft_flickr30k_qa](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_flickr30k_qa)
- [ro_sft_laion](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_laion)
- [ro_sft_llava_mix](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_llava_mix)
- [ro_sft_pixmo_aa](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_pixmo_aa)
- [ro_sft_pixmo_cap](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_pixmo_cap)
- [ro_sft_pixmo_cap_qa](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_pixmo_cap_qa)
- [ro_sft_pixmo_count](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_pixmo_count)
- [ro_sft_pixmo_points](https://huggingface.co/datasets/OpenLLM-Ro/ro_sft_pixmo_points)

Each Hugging Face dataset above ships **both** the **raw data** (which you can process locally with the scripts in this repo) and the **already-processed data**, ready to be fed directly to the model.

The complete set of registered datasets — including Romanian continued-pretraining, DPO, and text benchmarks (`ro_laroseda`, `ro_sts`, `ro_xquad`, `ro_wmt`) — is listed in [`data/dataset_info.json`](data/dataset_info.json).

> [!WARNING]
> **Local-data disclaimer.** The entries in [`data/dataset_info.json`](data/dataset_info.json) and the loader scripts under `data/` in this repo are wired to read from **local files/folders on disk**, *not* directly from the Hugging Face Hub. To train, download the relevant datasets (e.g. from [OpenLLM-Ro](https://huggingface.co/OpenLLM-Ro)) to local paths and adjust the entries in `dataset_info.json` to point at your copies. Large data files, tokenized caches, and checkpoints (`data/ro_vlm_*.json`, `data/lw/`, `saves/`, …) are **gitignored** and intentionally not shipped here.


![# LLaMA Factory](assets/logo.png)


- [Getting Started](#getting-started)




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


## Citation


```bibtex
@inproceedings{masala-etal-2024-vorbesti,
    title = "``Vorbe\c{s}ti Rom{\^a}ne\c{s}te?'' A Recipe to Train Powerful {R}omanian {LLM}s with {E}nglish Instructions",
    author = "Masala, Mihai and Ilie-Ablachim, Denis and Dima, Alexandru and Corlatescu, Dragos Georgian and Zavelca, Miruna-Andreea and Olaru, Ovio and Terian, Simina-Maria and Terian, Andrei and Leordeanu, Marius and Velicu, Horia and Popescu, Marius and Dascalu, Mihai and Rebedea, Traian",
    editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.681/",
    doi = "10.18653/v1/2024.findings-emnlp.681",
    pages = "11632--11647"
}

```

```bibtex
@misc{masala2026intelegi,
      title={``\^{I}n\c{t}elegi Rom\^{a}ne\c{s}te?'' A Recipe for Romanian Vision-Language Models},
      author={Mihai Masala and Marius Leordeanu and Mihai Dascalu and Traian Rebedea},
      year={2026},
      eprint={2605.31401},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.31401},
}
```

### Acknowledgement
This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank them for their wonderful work.


