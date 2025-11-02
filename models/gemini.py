from data.serialize import serialize_arr, SerializerSettings
import google.generativeai as genai
import os
import numpy as np
from jax import grad,vmap
from transformers import GPT2Tokenizer
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini API key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file or environment variables.")
genai.configure(api_key=api_key)

# Load a tokenizer to estimate token counts.
# Note: This is an approximation using GPT-2's tokenizer.
# For more accurate tokenization, a Gemini-specific tokenizer should be used if available.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_fn(s, model=None):
    """
    Tokenize a string to estimate token count.
    """
    return tokenizer.encode(s)

def get_allowed_ids(strs, model=None):
    """
    This function is kept for compatibility but is not used by the Gemini implementation
    as logit bias is not directly supported in the same way.
    """
    return []

def gemini_completion_fn(model, input_str, steps, settings, num_samples, temp):
    """
    Generate text completions from Gemini using Google's API, with a simple and robust prompt
    inspired by the successful structure in gpt.py.
    """
    generation_config = {
        "temperature": temp,
    }

    # # A simple, direct system instruction that works well for powerful models.
    # system_instruction = (
    #     "You are a helpful assistant that performs time series predictions. The user will provide a sequence "
    #     "and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
    # )

    # gemini_model = genai.GenerativeModel(
    #     model_name=model,
    #     system_instruction=system_instruction
    # )
    
    # # A clear and direct user prompt, proven to be effective.
    # # It explicitly asks the model to continue and specifies the output format.
    # prompt = (
    #     "Please continue the following sequence without producing any additional text. "
    #     "Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
    #     f"{input_str}{settings.time_sep}"
    # )

    # --- START: PROPOSED CHANGE (Explicit Length Instruction) ---
    # The 'steps' variable received here is actually expected_length * 1.2.
    # We can calculate the original length to be more precise.
    expected_length = int(steps / 1.2)

    system_instruction = (
        "You are a helpful assistant that performs time series predictions. The user will provide a sequence "
        "and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
    )

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction
    )
    
    # Add the exact number of steps to the prompt.
    prompt = (
        f"Please continue the following sequence for exactly {expected_length} more steps. "
        "Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
        f"{input_str}{settings.time_sep}"
    )
    # --- END: PROPOSED CHANGE ---

    responses = []
    print(f"Generating {num_samples} samples from Gemini ({model})...")
    for i in range(num_samples):
        print(f"  - Generating sample {i+1}/{num_samples}...")
        response = gemini_model.generate_content(prompt, generation_config=generation_config)
        try:
            # Clean up the response to remove potential markdown formatting
            clean_text = response.text.replace("```", "").strip()
            responses.append(clean_text)
        except ValueError:
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else 'UNKNOWN'
            print(f"Warning: Gemini API returned no content. Finish reason: {finish_reason}. Appending empty string.")
            responses.append("")
        
    return responses

def gemini_nll_fn(model, input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1):
    """
    Calculate the Negative Log-Likelihood (NLL) per dimension.
    
    NOTE: The Gemini API does not currently support obtaining log-probabilities for tokens in the same
    way as OpenAI's API. Therefore, this function is not implemented.
    """
    raise NotImplementedError("NLL calculation is not supported for Gemini models at this time.")
