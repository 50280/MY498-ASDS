
################## Overview ##################

# Aim: This file contians helper functions that are used 
# make calls to the different inference providers.
# This is part of the experimental section (study 2).  
# Date of last modification: 17.07.2025

##############################################

## Imports
import keyring
import time
import re # for regex detecting thinking tokens
import json  # for saving records to .jsonl
import pandas as pd  # assuming DataFrame rows are passed to build_record functions
import numpy as np   # for type handling like .item()
import requests  # if you later use it for Together or others
from openai import OpenAI  # for OpenAI API calls
import anthropic  # for Anthropic API client
import google.generativeai as genai # for gemini models through Google API


##############################################
# General
##############################################

# Function to save the record to .jsonl 
# This is used across all different model providers
# Using json lines allows to gradually build the datasets 
# and add each row as responses are generated without holding 
# the entire dataset in memory
def save_record_to_jsonl(record, path):
    """Appends a single record to a JSONL file. Converts any NumPy scalar 
    values in the record to native Python types and writes the result as 
    a JSON object on a new line in the specified file."""

    record = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
    with open(path, "a") as f:
        json.dump(record, f)
        f.write("\n")


# Function to build a record from the DataFrame row
# This helps structure the output from the API calls
def build_record(row, response_text, metadata, prompt_source):
    """Constructs a response record from the input row, model 
    output, and metadata. Takes a DataFrame row, generated response text, 
    metadata dictionary, and prompt source string; returns a dictionary 
    representing a complete record."""

    return {
        "utterance_id": row["utterance_id"],
        "user_prompt": row["user_prompt"],
        "counterfactual_prompt": row["counterfactual_prompt"],
        "prompt_template": row["prompt_template"],
        "whathow_q_debiased": int(row["whathow_q_debiased"]),
        "hobsons_c_debiased": int(row["hobsons_c_debiased"]),
        "model_response": response_text,
        "prompt_source": prompt_source,
        "model_metadata": metadata
    }

# Function for processing model responses; 
# extract thinking text from reasoning models. 
def extract_reasoning_and_strip_all(text):
    """Extract reasoning inside <think>...</think> tags, 
    return reasoning and stripped text."""
    if not isinstance(text, str):
        return pd.Series([np.nan, text])
    
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    reasoning_combined = "\n\n---\n\n".join(m.strip() for m in matches) if matches else np.nan

    # Remove all well-formed <think>...</think> blocks
    stripped_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any leftover malformed <think> or </think> tags
    stripped_text = re.sub(r"</?think>", "", stripped_text, flags=re.DOTALL).strip()

    return pd.Series([reasoning_combined, stripped_text])

##############################################
# Open AI
##############################################

# Function to call OpenAI with a given prompt
def openai_call_model(prompt, model):
    """Sends a prompt to the OpenAI Chat API and retrieves the 
    generated response along with usage metadata. Takes a prompt string 
    and model name; returns the response text and a dictionary containing 
    model info, token usage, and response metadata."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    reply_text = response.choices[0].message.content
    metadata = {
        "model": response.model,
        "response_id": response.id,
        "created": response.created,
        "finish_reason": response.choices[0].finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }
    return reply_text, metadata


##############################################
# Anthropic 
##############################################

# Function to call Anthropic with a given prompt
def anthropic_call_model(prompt, model):
    """Sends a prompt to the Anthropic API and retrieves the model's 
    generated response and metadata. Takes a prompt string and model name; 
    returns the response text and a dictionary with metadata including usage 
    and stop reason."""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    reply_text = response.content[0].text
    metadata = {
        "model": response.model,
        "response_id": response.id,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    }
    return reply_text, metadata


##############################################
# Mistral AI
##############################################

# Function to call the Mistral model
def mistralai_call_model(prompt, model, url, headers):
    """Sends a prompt to the Mistral API and retrieves the model's 
    response and associated metadata. Takes a prompt string, model name, 
    API URL, and request headers; returns the generated reply text and a metadata dictionary."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
        # No temperature or max_tokens set — using API defaults
    }

    response = requests.post(url, headers=headers, json=payload)
    if not response.ok:
        raise Exception(f"Mistral API error: {response.status_code} - {response.text}")

    data = response.json()
    reply_text = data["choices"][0]["message"]["content"]
    metadata = {
        "model": data.get("model"),
        "response_id": data.get("id"),
        "created": data.get("created"),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "usage": data.get("usage")
    }
    return reply_text, metadata


##############################################
# Together AI (Provider to access Deepseek & Meta Models) 
##############################################

# This one is formatted with some setups. 
api_key = keyring.get_password("together", "capstone_collect_responses")

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Function to call model through Together AI API with a given prompt
def together_call_model(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """Sends a prompt to the Together API to generate a model response 
    and extract metadata. Takes a prompt string and optional model name; 
    returns the generated reply text and a metadata dictionary."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
        # No temperature, no max_tokens - use default for each model
    }

    response = requests.post(TOGETHER_API_URL, headers=HEADERS, json=payload)
    if not response.ok:
        raise Exception(f"Together API error: {response.status_code} - {response.text}")

    data = response.json()
    reply_text = data["choices"][0]["message"]["content"]
    metadata = {
        "model": data.get("model"),
        "response_id": data.get("id"),
        "created": data.get("created"),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "usage": data.get("usage")
    }
    return reply_text, metadata


##############################################
# Google API (for gemini models)
##############################################

def google_call_model(prompt_text, model):
    """
    Calls Gemini model and returns response and metadata.
    """
    try:
        response = model.generate_content(prompt_text)
        response_text = response.text

        metadata = {
            "model": model._model_name,
            "safety_ratings": response.prompt_feedback.safety_ratings
                if hasattr(response, "prompt_feedback") and response.prompt_feedback
                else None
        }

        return response_text, metadata
    except Exception as e:
        raise RuntimeError(f"Google API call failed: {e}")



##############################################
# Perspective API - to get bridging attributes
##############################################

# Perspective API call
def call_perspective(text, requested_attributes, api_url, retries=3, sleep=1):
    """Sends a text input to the Perspective API and retrieves 
    bridging-related attribute scores. Takes a text string and optional 
    retry/sleep settings; returns a dictionary of attribute scores or an 
    empty dict on failure."""

    if not isinstance(text, str) or not text.strip():
        return {}
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": requested_attributes
    }
    for attempt in range(retries):
        try:
            response = requests.post(api_url, json=data)
            response.raise_for_status()
            return response.json().get("attributeScores", {})
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(sleep)
            else:
                print(f"Error for text: {text[:30]}... → {e}")
                return {}

# Function to flatten the output scores 
def flatten_scores(score_dict, requested_attributes):
    """ Flattens a nested dictionary of attribute scores by 
    extracting the summary score value for each requested attribute.
    Returns a new dictionary with lowercase attribute names as keys and 
    their corresponding summary score values or None if unavailable."""

    return {
        key.lower(): score_dict.get(key, {}).get("summaryScore", {}).get("value", None)
        for key in requested_attributes.keys()
    }