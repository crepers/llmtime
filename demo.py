import os
import torch
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
import google.generativeai as genai

# Conditionally initialize the LLM provider
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'gpt').lower()
if LLM_PROVIDER == 'gpt':
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the .env file or environment variables.")
    openai.api_key = api_key
    openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
elif LLM_PROVIDER == 'gemini':
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the .env file or environment variables.")
    genai.configure(api_key=api_key)
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data

def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()



print(torch.cuda.max_memory_allocated())
print()

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

mistral_api_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)

gemini_flash_hypers = dict(
    temp=0.3, # Keep lower temperature for stability
    alpha=0.3,  # Align with gpt4_hypers
    basic=True, # Align with gpt4_hypers
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-') # Fully align with gpt4_hypers
)

gemini_pro_hypers = dict(
    temp=0.3, # Keep lower temperature for stability
    alpha=0.3,  # Align with gpt4_hypers
    basic=True, # Align with gpt4_hypers
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-') # Fully align with gpt4_hypers
)


promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)

arima_hypers = dict(p=[12,30], d=[1,2], q=[0])

# Define model configurations based on the selected provider
if LLM_PROVIDER == 'gpt':
    print("Using GPT models for testing.")
    model_hypers = {
        'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
        'LLMTime GPT-4': {'model': 'gpt-4', **gpt4_hypers},
        'LLMTime GPT-3': {'model': 'text-davinci-003', **gpt3_hypers},
        'PromptCast GPT-3': {'model': 'text-davinci-003', **promptcast_hypers},
        'ARIMA': arima_hypers,
    }
    model_predict_fns = {
        'LLMTime GPT-3.5': get_llmtime_predictions_data,
        'LLMTime GPT-4': get_llmtime_predictions_data,
        'LLMTime GPT-3': get_llmtime_predictions_data,
        'PromptCast GPT-3': get_promptcast_predictions_data,
        'ARIMA': get_arima_predictions_data,
    }
elif LLM_PROVIDER == 'gemini':
    print("Using Gemini models for testing.")
    model_hypers = {
        'LLMTime Gemini Flash 2.5': {'model': 'gemini-2.5-flash', **gemini_flash_hypers},
        'LLMTime Gemini 2.5 Pro': {'model': 'gemini-2.5-pro', **gemini_pro_hypers},
        'ARIMA': arima_hypers,
    }
    model_predict_fns = {
        'LLMTime Gemini Flash 2.5': get_llmtime_predictions_data,
        'LLMTime Gemini 2.5 Pro': get_llmtime_predictions_data,
        'ARIMA': get_arima_predictions_data,
    }
elif LLM_PROVIDER == 'mistral':
    print("Using Mistral models for testing.")
    model_hypers = {
        'mistral-api-tiny': {'model': 'mistral-api-tiny', **gemini_hypers}, # Using gemini_hypers as a stand-in
        'ARIMA': arima_hypers,
    }
    model_predict_fns = {
        'mistral-api-tiny': get_llmtime_predictions_data,
        'ARIMA': get_arima_predictions_data,
    }
else:
    print(f"Unknown or unconfigured LLM_PROVIDER: '{LLM_PROVIDER}'. Running only ARIMA model.")
    model_hypers = {
        'ARIMA': arima_hypers,
    }
    model_predict_fns = {
        'ARIMA': get_arima_predictions_data,
    }

model_names = list(model_predict_fns.keys())

datasets = get_datasets()
ds_name = 'AirPassengersDataset'


data = datasets[ds_name]
train, test = data # or change to your own data
out = {}
for model in model_names: # GPT-4 takes a about a minute to run
    print(f"--- Running model: {model} ---")
    # Safe logging: Check if train object has an index before printing
    if hasattr(train, 'index'):
        print(f"Before execution, train data index ends with: {train.index[-5:]}...")
    else:
        print(f"Before execution, train data (last 5): {train[-5:]}...")

    if 'PromptCast' in model:
        model_hypers[model].update({'dataset_name': ds_name}) # for promptcast
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 10
    # CRITICAL FIX: Pass copies of the data to the prediction function
    # This protects the original train/test data from being modified by any model (like ARIMA).
    pred_dict = get_autotuned_predictions_data(train.copy(), test.copy(), hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
    out[model] = pred_dict
    
    # Safe logging: Check if train object has an index before printing
    if hasattr(train, 'index'):
        print(f"After execution, train data index ends with: {train.index[-5:]}...")
    else:
        print(f"After execution, train data (last 5): {train[-5:]}...")

    plot_preds(train, test, pred_dict, model, show_samples=True)
    print("-" * (20 + len(model)))