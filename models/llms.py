import os
from functools import partial
from models.gpt import gpt_completion_fn, gpt_nll_fn
from models.gpt import tokenize_fn as gpt_tokenize_fn
from models.llama import llama_completion_fn, llama_nll_fn
from models.llama import tokenize_fn as llama_tokenize_fn

from models.mistral import mistral_completion_fn, mistral_nll_fn
from models.mistral import tokenize_fn as mistral_tokenize_fn

from models.mistral_api import mistral_api_completion_fn, mistral_api_nll_fn
from models.mistral_api import tokenize_fn as mistral_api_tokenize_fn

# Import Gemini functions
from models.gemini import gemini_completion_fn, gemini_nll_fn
from models.gemini import tokenize_fn as gemini_tokenize_fn


# Determine the LLM provider from environment variables
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'gpt').lower()

# Initialize dictionaries
completion_fns = {}
nll_fns = {}
tokenization_fns = {}
context_lengths = {}

if LLM_PROVIDER == 'gpt':
    # GPT models
    completion_fns.update({
        'text-davinci-003': partial(gpt_completion_fn, model='text-davinci-003'),
        'gpt-4': partial(gpt_completion_fn, model='gpt-4'),
        'gpt-4-1106-preview': partial(gpt_completion_fn, model='gpt-4-1106-preview'),
        'gpt-3.5-turbo-instruct': partial(gpt_completion_fn, model='gpt-3.5-turbo-instruct'),
    })
    nll_fns.update({
        'text-davinci-003': partial(gpt_nll_fn, model='text-davinci-003'),
    })
    tokenization_fns.update({
        'text-davinci-003': partial(gpt_tokenize_fn, model='text-davinci-003'),
        'gpt-3.5-turbo-instruct': partial(gpt_tokenize_fn, model='gpt-3.5-turbo-instruct'),
    })
    context_lengths.update({
        'text-davinci-003': 4097,
        'gpt-3.5-turbo-instruct': 4097,
    })
elif LLM_PROVIDER == 'gemini':
    # Gemini models
    GEMINI_FLASH_2_5 = 'gemini-2.5-flash' # A fast and capable model
    GEMINI_PRO_2_5 = 'gemini-2.5-pro'     # A stronger, more advanced model
    
    completion_fns.update({
        GEMINI_FLASH_2_5: partial(gemini_completion_fn, model=GEMINI_FLASH_2_5),
        GEMINI_PRO_2_5: partial(gemini_completion_fn, model=GEMINI_PRO_2_5),
    })
    nll_fns.update({
        # Gemini models do not support NLL calculation
    })
    tokenization_fns.update({
        GEMINI_FLASH_2_5: partial(gemini_tokenize_fn, model=GEMINI_FLASH_2_5),
        GEMINI_PRO_2_5: partial(gemini_tokenize_fn, model=GEMINI_PRO_2_5),
    })
    context_lengths.update({
        GEMINI_FLASH_2_5: 8192, # Assuming same context length for simplicity
        GEMINI_PRO_2_5: 8192,
    })
elif LLM_PROVIDER == 'mistral':
    # Mistral models
    completion_fns.update({
        'mistral': partial(mistral_completion_fn, model='mistral'),
        'mistral-api-tiny': partial(mistral_api_completion_fn, model='mistral-tiny'),
        'mistral-api-small': partial(mistral_api_completion_fn, model='mistral-small'),
        'mistral-api-medium': partial(mistral_api_completion_fn, model='mistral-medium'),
    })
    nll_fns.update({
        'mistral': partial(mistral_nll_fn, model='mistral'),
        'mistral-api-tiny': partial(mistral_api_nll_fn, model='mistral-tiny'),
        'mistral-api-small': partial(mistral_api_nll_fn, model='mistral-small'),
        'mistral-api-medium': partial(mistral_api_nll_fn, model='mistral-medium'),
    })
    tokenization_fns.update({
        'mistral': partial(mistral_tokenize_fn, model='mistral'),
        'mistral-api-tiny': partial(mistral_api_tokenize_fn, model='mistral-tiny'),
        'mistral-api-small': partial(mistral_api_tokenize_fn, model='mistral-small'),
        'mistral-api-medium': partial(mistral_api_tokenize_fn, model='mistral-medium'),
    })
    context_lengths.update({
        'mistral-api-tiny': 4097,
        'mistral-api-small': 4097,
        'mistral-api-medium': 4097,
        'mistral': 4096,
    })
elif LLM_PROVIDER == 'llama':
    # Llama models
    completion_fns.update({
        'llama-7b': partial(llama_completion_fn, model='7b'),
        'llama-13b': partial(llama_completion_fn, model='13b'),
        'llama-70b': partial(llama_completion_fn, model='70b'),
        'llama-7b-chat': partial(llama_completion_fn, model='7b-chat'),
        'llama-13b-chat': partial(llama_completion_fn, model='13b-chat'),
        'llama-70b-chat': partial(llama_completion_fn, model='70b-chat'),
    })
    nll_fns.update({
        'llama-7b': partial(llama_nll_fn, model='7b'),
        'llama-13b': partial(llama_nll_fn, model='13b'),
        'llama-70b': partial(llama_nll_fn, model='70b'),
        'llama-7b-chat': partial(llama_nll_fn, model='7b-chat'),
        'llama-13b-chat': partial(llama_nll_fn, model='13b-chat'),
        'llama-70b-chat': partial(llama_nll_fn, model='70b-chat'),
    })
    tokenization_fns.update({
        'llama-7b': partial(llama_tokenize_fn, model='7b'),
        'llama-13b': partial(llama_tokenize_fn, model='13b'),
        'llama-70b': partial(llama_tokenize_fn, model='70b'),
        'llama-7b-chat': partial(llama_tokenize_fn, model='7b-chat'),
        'llama-13b-chat': partial(llama_tokenize_fn, model='13b-chat'),
        'llama-70b-chat': partial(llama_tokenize_fn, model='70b-chat'),
    })
    context_lengths.update({
        'llama-7b': 4096,
        'llama-13b': 4096,
        'llama-70b': 4096,
        'llama-7b-chat': 4096,
        'llama-13b-chat': 4096,
        'llama-70b-chat': 4096,
    })
