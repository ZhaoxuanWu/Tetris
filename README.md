# <img src="tetris.png" alt="Tetris" style="height:1em; vertical-align:middle;"> TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding
> 🔬 **ACL 2025 (main)** | [paper link](https://arxiv.org/pdf/2502.15197)  
> **Zhaoxuan Wu^, Zijian Zhou^, Arun Verma, Alok Prakash, Daniela Rus, and Bryan Kian Hsiang Low**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
![GitHub Repo stars](https://img.shields.io/github/stars/ZhaoxuanWu/Tetris)

⭐️ Like this work? Give us a star~

---

# What is Tetris <img src="tetris.png" alt="Tetris" style="height:1em; vertical-align:middle;">?

**TETRIS** optimizes the total throughput of batch speculative decoding in multi-request settings by strategically selecting draft tokens for verification.

> ⭐ **_Fast LLM inference for service providers with limited inference capacity._**  
> ⏩ Designed to **maximize throughput** and **minimize wasted computation.**  
> 🚀 Ideal for **LLM service providers with limited compute resources.**

Unlike existing methods that optimize for a single request or a group of requests as a whole, Tetris actively selects the most promising draft tokens (for every request in a batch) to be accepted when verified in parallel, resulting in fewer rejected tokens and hence less wasted computing resources.

<!-- Such an effective resource utilization to achieve fast inference in large language models (LLMs) is especially important to service providers with limited inference capacity. -->

<img src="tetris_fig.png" alt="Tetris Figure" style="max-width:600px; width:100%;">

# 📦 Getting Started

Install the library from source.
```bash
export MAX_JOBS=6
conda install ccache
pip install -e .
```
The build will take a few minutes to complete. If you encounter any problem, please refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html) for a complete guide.

Install *pytorch_scatter* binaries for PyTorch 2.6.0. Other installation methods can be found [here](https://github.com/rusty1s/pytorch_scatter).
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+${CUDA}.html
```

Install the up-to-date FastChat library from source.
```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install --upgrade pip
pip3 install -e .
```

# Prepare the data
You can download the dataset by running:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

You can get the Arena dataset and the Domain-Specific Tough Questions dataset by running [`convert_datasets.py`](convert_datasets.py):
```bash
python convert_datasets.py
```

# Run our code
The main logic for Tetris is in [`vllm/spec_decode/tetris.py`](vllm/spec_decode/tetris.py).

The configurations for the experiments are in [`benchmarks/dsd/scripts/run_tetris.sh`](benchmarks/dsd/scripts/run_tetris.sh).

To run experiments, execute the following command:
```bash
bash benchmarks/dsd/scripts/run_tetris.sh
```

# 📝 Citation
If you have found our work interesting and have used it in your own project/research, kindly cite us:
```bibtex
@inproceedings{wu2024tetris,
  title={TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding},
  author={Wu, Zhaoxuan and Zhou, Zijian and Verma, Arun and Prakash, Alok and Rus, Daniela and Low, Bryan Kian Hsiang},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2025}
}
```

# Note and acknowledgments
This project builds upon the work of others. 
Our code is a fork from https://github.com/LiuXiaoxuanPKU/vllm.
It contains the original contents from the [vLLM library](https://github.com/vllm-project/vllm), specifically version v0.4.2.
We thank the contributors of the vLLM project and Xiaoxuan Liu for their amazing implementation.
