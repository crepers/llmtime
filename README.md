# Large Language Models Are Zero Shot Time Series Forecasters

This repository contains the code for the paper
[_Large Language Models Are Zero Shot Time Series Forecasters_](https://arxiv.org/abs/2310.07820)
by Nate Gruver, Marc Finzi, Shikai Qiu and Andrew Gordon Wilson (NeurIPS 2023).

<figure>
  <img src="./assets/llmtime_top_fig.png" alt="Image">
  <figcaption> We propose <em>LLMTime</em>, a method for <em>zero-shot</em> time series forecasting with large language models (LLMs) by encoding numbers as text and sampling possible extrapolations as text completions. LLMTime can outperform many popular timeseries methods without any training on the target dataset (i.e. zero shot). The performance of LLMTime also scales with the power of the underlying base model. However, models that undergo alignment (e.g. RLHF) do not follow the scaling trend. For example, GPT-4 demonstrates inferior performance to GPT-3. </figcaption>
</figure>

## 🛠 Installation
First, create a `.env` file by copying the example and edit it to add your API keys:
```bash
cp .env.sample .env
```
Inside the `.env` file, set the `LLM_PROVIDER` to either `gpt` or `gemini` and provide the corresponding API keys (e.g., `OPENAI_API_KEY` or `GEMINI_API_KEY`).

Next, install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```
It is recommended to use a virtual environment. For example, with conda:
```bash
conda create -n llmtime python=3.10
conda activate llmtime
pip install -r requirements.txt
```

## 🚀 Trying out LLMTime
Want a quick taste of the power of LLMTime? Run the quick demo in the `demo.ipynb` notebook or the `demo.py` script.

Make sure your `.env` file is configured correctly. You can switch between different LLM providers like GPT and Gemini by changing the `LLM_PROVIDER` variable in the `.env` file.

## ✨ Using TimesFM
This project also supports [Google's TimesFM](https://github.com/google-research/timesfm), a powerful, pretrained time-series foundation model.

**Requirements:**
- To use TimesFM, you must have **Python 3.10 or higher**.

**Installation:**
You can install the official `timesfm` package along with its dependencies using the following command, as recommended on their [Hugging Face page](https://huggingface.co/google/timesfm-2.5-200m-pytorch):
```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .
```
Note: This is already included in the `requirements.txt` file.

## 🤖 Plugging in other LLMs
We currently support GPT-3, GPT-3.5, GPT-4, Gemini (Flash and Pro), TimesFM, Mistral, and LLaMA 2. You can easily plug in other LLMs by adding a completion function in `models/` and registering it in `models/llms.py`.

You can also experiment with different settings for each model by modifying the `*_hypers` dictionaries in `demo.ipynb` or `demo.py`.

## 💡 Tips 
Here are some tips for using LLMTime:
- Performance is not too sensitive to the data scaling hyperparameters `alpha, beta, basic`. A good default is `alpha=0.95, beta=0.3, basic=False`. For data exhibiting symmetry around 0 (e.g. a sine wave), we recommend setting `basic=True` to avoid shifting the data.
- The recently released `gpt-3.5-turbo-instruct` seems to require a lower temperature (e.g. 0.3) than other models, and tends to not outperform `text-davinci-003` from our limited experiments.
- Tuning hyperparameters based on validation likelihoods, as done by `get_autotuned_predictions_data`, will often yield better test likelihoods, but won't necessarily yield better samples. 

## 📊 Replicating experiments in paper
Run the following commands to replicate the experiments in the paper. The outputs will be saved in `./outputs/`. You can use `visualize.ipynb` to visualize the results. We also provide precomputed outputs used in the paper in `./precomputed_outputs/`.
### Darts (Section 4)
```
python -m experiments.run_darts
```
### Monash (Section 4)
You can download preprocessing data from [here](https://drive.google.com/file/d/1sKrpWbD3LvLQ_e5lWgX3wJqT50sTd1aZ/view?usp=sharing) or use the following command
```
gdown 'https://drive.google.com/uc?id=1sKrpWbD3LvLQ_e5lWgX3wJqT50sTd1aZ'
```
Then extract the data (the extracted data will be in `./datasets/monash/`)
```
tar -xzvf monash.tar.gz
```
Then run the experiment
```
python -m experiments.run_monash
```
### Synthetic (Section 5)
```
python -m experiments.run_synthetic
```
### Missing values (Section 6)
```
python -m experiments.run_missing
```
### Memorization (Appendix B)
```
python -m experiments.run_memorization
```

## Citation
Please cite our work as:
```bibtex
@inproceedings{gruver2023llmtime,
    title={{Large Language Models Are Zero Shot Time Series Forecasters}},
    author={Nate Gruver, Marc Finzi, Shikai Qiu and Andrew Gordon Wilson},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023}
}
```
