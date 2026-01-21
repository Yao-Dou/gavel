"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
from typing import Any, Union, List, Dict

import json
import re

import aiolimiter

import openai
from openai import AsyncAzureOpenAI, AzureOpenAI, OpenAI, AsyncOpenAI
import anthropic
from anthropic import AsyncAnthropic
from mistralai import Mistral, models
from google import genai
from google.genai import types, errors
import pydantic

from tqdm.asyncio import tqdm_asyncio

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.aio import ChatCompletionsClient as AsyncChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

import copy

from dotenv import load_dotenv
dotenv_path = os.path.expanduser('/srv/nlprx-lab/share6/douy/common.env')
load_dotenv(dotenv_path)
os.environ['CURL_CA_BUNDLE'] = ''

anthropic_client = anthropic.Anthropic()


ERROR_ERRORS_TO_MESSAGES = {
    openai.UnprocessableEntityError: "OpenAI API Invalid Request: Prompt was filtered",
    openai.RateLimitError: "OpenAI API rate limit exceeded. Sleeping for 10 seconds.",
    openai.APIStatusError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    openai.APITimeoutError: "OpenAI APITimeout Error: OpenAI Timeout",
    openai.InternalServerError: "OpenAI service unavailable error: {e}",
    openai.APIError: "OpenAI API error: {e}",
    openai.APIConnectionError: "OpenAI API Connection error: {e}",
    openai.BadRequestError: "OpenAI API Bad Request error: {e}",
}

import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

import os
import json


def merge_nested_dicts(dict1, dict2):
    """
    Merges two nested dictionaries with unique ending keys.
    Args:
        dict1: First dictionary
        dict2: Second dictionary
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    def recursive_merge(current_dict, other_dict):
        for key, value in other_dict.items():
            if key not in current_dict:
                current_dict[key] = value
            elif isinstance(value, dict) and isinstance(current_dict[key], dict):
                recursive_merge(current_dict[key], value)
            else:
                # If we reach here, we're at a leaf node or there's a conflict
                # Since we guarantee unique ending keys, we keep the existing value
                print(f"Conflict at key: {key}. Keeping existing value.")
                continue
    
    recursive_merge(result, dict2)
    return result

def num_tokens_per_string(s):
    return len(enc.encode(s))

async def _throttled_openai_chat_completion_acreate(
    client: Union[AsyncAzureOpenAI, AsyncOpenAI],
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    json_mode: bool,
    limiter: aiolimiter.AsyncLimiter,
    reasoning_effort: str = "medium",
) -> dict[str, Any]:

    # o3 model does not support temperature and top_p change
    if model == "o3":
        top_p = 1.0
        temperature = 1.0

    async with limiter:
        for _ in range(20):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "top_p": top_p,
                    "n": n,
                }
                if model == "o3":
                    kwargs["reasoning_effort"] = reasoning_effort
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                return await client.chat.completions.create(**kwargs)
            except Exception as e:
                if isinstance(e, openai.UnprocessableEntityError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": ""
                                }
                            }
                            for _ in range(n)
                        ]
                    }
                elif isinstance(e, openai.BadRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": ""
                                }
                            }
                            for _ in range(n)
                        ]
                    }
                # else:
                #     logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

async def generate_from_openai_chat_completion(
    full_contexts: list,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    n: int = 1,
    json_mode: bool = False,
    requests_per_minute: int = 200,
    show_progress: bool = True,
    reasoning_effort: str = "medium",
) -> list[list[str]]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        model_name: Model name.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """

    client = AsyncOpenAI(
        api_key = os.environ.get("OPENAI_API_KEY_YAO"),
    )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client=client,
            model=model_name,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            json_mode=json_mode,
            limiter=limiter,
            reasoning_effort=reasoning_effort,
        )
        for full_context in full_contexts
    ]

    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    return responses

async def generate_from_azure_openai_chat_completion(
    full_contexts: list,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    n: int = 1,
    json_mode: bool = False,
    requests_per_minute: int = 100,
    show_progress: bool = True,
    max_concurrent: int = 100,
    reasoning_effort: str = "medium",
) -> list[list[str]]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        model_name: Model name.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """

    # client = AsyncOpenAI(
    #     api_key = os.environ.get("OPENAI_API_KEY_YAO"),
    # )

    client = AsyncAzureOpenAI(
        api_version="2025-01-01-preview",
        azure_endpoint="https://yao-inference2.openai.azure.com/",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY_XLAB"),
    )

    if model_name not in ["o3-2025-04-16", "gpt-4.1-2025-04-14"]:
        raise ValueError(f"Model {model_name} is not supported")

    if model_name == "o3-2025-04-16":
        model_name = "o3"
    elif model_name == "gpt-4.1-2025-04-14":
        model_name = "gpt-4.1"

    limiter = aiolimiter.AsyncLimiter(requests_per_minute, time_period=60)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_task(context):
        # Only allow max_concurrent tasks to run concurrently.
        async with semaphore:
            return await _throttled_openai_chat_completion_acreate(
                client=client,
                model=model_name,
                messages=context,
                temperature=temperature if temperature is not None else 0,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                json_mode=json_mode,
                limiter=limiter,
                reasoning_effort=reasoning_effort,
            )

    # Create a task for each context.
    async_responses = [limited_task(context) for context in full_contexts]

    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    return responses

ANTHROPIC_ERROR_MESSAGES = {
    anthropic.APIConnectionError: "Anthropic API Connection Error: Could not reach server",
    anthropic.RateLimitError: "Anthropic API Rate Limit Error: Backing off",
    anthropic.BadRequestError: "Anthropic API Bad Request Error: {e}",
    anthropic.AuthenticationError: "Anthropic API Authentication Error: Invalid API key",
    anthropic.PermissionDeniedError: "Anthropic API Permission Error: {e}",
    anthropic.NotFoundError: "Anthropic API Not Found Error: {e}",
    anthropic.UnprocessableEntityError: "Anthropic API Unprocessable Entity Error: {e}",
    anthropic.InternalServerError: "Anthropic API Internal Server Error: {e}",
    anthropic.APIStatusError: "Anthropic API Status Error: Non-200 status code received {e}"
}

async def _throttled_anthropic_chat_completion_acreate(
    client: AsyncAnthropic,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    limiter: aiolimiter.AsyncLimiter,
    thinking_enabled: bool = False,
    thinking_budget: int = 10000,
) -> Dict[str, Any]:
    """Throttled async chat completion for Anthropic API with error handling and retries."""

    # Make a deep copy of messages to avoid changing the original object
    messages_copy = copy.deepcopy(messages)

    system_prompt_cached = [
        {"type": "text", "text": messages_copy[0]["content"], "cache_control": {"type": "ephemeral"}},
    ]
    if messages_copy[-1]["role"] == "user":
        messages_copy[-1]["content"] = [
            {
                "type": "text",
                "text": messages_copy[-1]["content"],
                "cache_control": {"type": "ephemeral"}
            }
        ]

    async with limiter:
        for attempt in range(20):  # Retry logic
            try:
                responses = []
                # Anthropic doesn't support 'n' directly, so we make multiple calls
                for _ in range(n):
                    # Use temperature 1.0 for thinking models
                    adjusted_temperature = temperature
                    if thinking_enabled:
                        adjusted_temperature = 1.0
                    
                    # Build common parameters
                    params = {
                        "model": model,
                        "extra_headers": {
                            "anthropic-beta": "prompt-caching-2024-07-31"
                        },
                        "max_tokens": max_tokens,
                        "temperature": adjusted_temperature if adjusted_temperature is not None else 0,
                    }
                    
                    # Claude 4 models don't support both temperature and top_p
                    # Only add top_p for older models
                    if model not in ["claude-sonnet-4-20250514", "claude-opus-4-1-20250805", "claude-opus-4-20250514"]:
                        params["top_p"] = top_p if top_p is not None else 1.0
                    
                    # Add thinking parameter if enabled
                    if thinking_enabled:
                        params["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": thinking_budget
                        }
                    
                    if messages_copy[0]["role"] == "system":
                        response = await client.messages.create(
                            messages=messages_copy[1:],  # Exclude system message from messages
                            system=system_prompt_cached,
                            **params
                        )
                    else:
                        response = await client.messages.create(
                            messages=messages_copy,
                            **params
                        )
                    responses.append(response)
                
                # Extract text content from responses, handling thinking blocks
                formatted_responses = []
                for resp in responses:
                    # Extract only text content blocks
                    text_content = ""
                    for block in resp.content:
                        if hasattr(block, 'text'):  # Text block
                            text_content += block.text
                    formatted_responses.append({
                        "message": {
                            "content": text_content
                        }
                    })
                
                return {
                    "choices": formatted_responses
                }

            except tuple(ANTHROPIC_ERROR_MESSAGES.keys()) as e:
                if isinstance(e, anthropic.RateLimitError):
                    # logging.warning(f"{ANTHROPIC_ERROR_MESSAGES[type(e)]}. Waiting {10} seconds.")
                    await asyncio.sleep(10)
                    continue
                
                elif isinstance(e, anthropic.APIConnectionError):
                    # logging.warning(f"{ANTHROPIC_ERROR_MESSAGES[type(e)]}: {e.__cause__}")
                    await asyncio.sleep(10)
                    continue
                
                elif isinstance(e, anthropic.InternalServerError):
                    # logging.warning(ANTHROPIC_ERROR_MESSAGES[type(e)].format(e=e))
                    await asyncio.sleep(10)
                    continue
                
                elif isinstance(e, (anthropic.BadRequestError, 
                                 anthropic.UnprocessableEntityError)):
                    logging.warning(ANTHROPIC_ERROR_MESSAGES[type(e)].format(e=e))
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": ""
                                }
                            } for _ in range(n)
                        ]
                    }
                
                elif isinstance(e, (anthropic.AuthenticationError, 
                                 anthropic.PermissionDeniedError)):
                    logging.error(ANTHROPIC_ERROR_MESSAGES[type(e)].format(e=e))
                    raise e
                
                else:
                    # logging.warning(ANTHROPIC_ERROR_MESSAGES[type(e)].format(e=e))
                    await asyncio.sleep(10)
                    
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

async def generate_from_anthropic_chat_completion(
    full_contexts: List[List[Dict[str, str]]],
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    n: int = 1,
    requests_per_minute: int = 100,
    show_progress: bool = True,
    max_concurrent: int = 100,
) -> List[Dict[str, Any]]:
    """Generate from Anthropic Chat Completion API.

    Args:
        full_contexts: List of message lists to generate from.
        model_name: Model name (default: "claude-3-opus-20240229").
        temperature: Temperature for generation (0-1).
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p sampling parameter.
        n: Number of responses to generate for each API call.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses in OpenAI-like format.
    """
    client = AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_KEY"),
    )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Check if thinking mode is enabled
    thinking_enabled = model_name.endswith("-thinking")
    actual_model = model_name[:-len("-thinking")] if thinking_enabled else model_name
    thinking_budget = 16000 if thinking_enabled else None

    async def limited_task(context):
        # Only allow max_concurrent tasks to run concurrently.
        async with semaphore:
            return await _throttled_anthropic_chat_completion_acreate(
                client=client,
                model=actual_model,
                messages=context,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                limiter=limiter,
                thinking_enabled=thinking_enabled,
                thinking_budget=thinking_budget,
            )

    # Create a task for each context.
    async_responses = [limited_task(context) for context in full_contexts]
    
    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    return responses

# Define error messages for Gemini API
GEMINI_ERROR_MESSAGES = {
    errors.APIError: "Gemini API Error: {e}",
    errors.ClientError: "Gemini API Client Error: {e}",
    errors.ServerError: "Gemini API Server Error: {e}",
    errors.UnknownFunctionCallArgumentError: "Gemini API Unknown Function Call Argument Error: {e}",
    errors.UnsupportedFunctionError: "Gemini API Unsupported Function Error: {e}",
    errors.FunctionInvocationError: "Gemini API Function Invocation Error: {e}",
    pydantic.ValidationError: "Gemini API Validation Error: {e}"
}

async def _throttled_gemini_generate_content_acreate(
    client: genai.Client,
    model: str,
    contents: List[types.Content],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    thinking_budget: int,
    system_prompt: str,
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    """Throttled async content generation for Gemini API with error handling and retries."""
    
    # Create the generation config
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature if temperature is not None else 0,
        top_p=top_p if top_p is not None else 1.0,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
        candidate_count=n,
        system_instruction=system_prompt if system_prompt else None,
        thinking_config = types.ThinkingConfig(
            thinking_budget=thinking_budget,
        ) if thinking_budget is not None else None,
    )
    
    async with limiter:
        for attempt in range(20):  # Retry logic
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                # return response

                # TODO: Change Back
                # Format response to match OpenAI-like structure for consistency
                return {
                    "choices": [
                        {
                            "message": {
                                "content": candidate.content.parts[0].text
                            }
                        }
                        for candidate in response.candidates
                    ]
                }
                
            except errors.APIError as e:
                # Handle different error types based on error code
                if hasattr(e, 'code'):
                    if e.code == 429:  # RESOURCE_EXHAUSTED
                        # Rate limit exceeded
                        await asyncio.sleep(10)
                        continue
                        
                    elif e.code in [500, 503, 504]:  # INTERNAL, UNAVAILABLE, DEADLINE_EXCEEDED
                        # Server errors, retry after delay
                        await asyncio.sleep(10)
                        continue
                        
                    elif e.code in [400, 404]:  # INVALID_ARGUMENT, NOT_FOUND
                        # Client errors, log warning and return empty response
                        logging.warning(GEMINI_ERROR_MESSAGES[type(e)].format(e=e))
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": ""
                                    }
                                } for _ in range(n)
                            ]
                        }
                        
                    elif e.code == 403:  # PERMISSION_DENIED
                        # Authentication errors, log error and raise exception
                        logging.error(GEMINI_ERROR_MESSAGES[type(e)].format(e=e))
                        return {
                            "choices": [
                               {
                                    "message": {
                                        "content": ""
                                    }
                                } for _ in range(n)
                            ] 
                        }
                
                # General error handling if code attribute is not available
                logging.warning(GEMINI_ERROR_MESSAGES[type(e)].format(e=e))
                await asyncio.sleep(10)
                continue
                    
            except (errors.ClientError, errors.ServerError, 
                    errors.UnknownFunctionCallArgumentError,
                    errors.UnsupportedFunctionError, 
                    errors.FunctionInvocationError,
                    pydantic.ValidationError) as e:
                # Log warning and return empty response immediately
                logging.warning(GEMINI_ERROR_MESSAGES[type(e)].format(e=e))
                return {
                    "choices": [
                        {
                            "message": {
                                "content": ""
                            }
                        } for _ in range(n)
                    ]
                }

            except Exception as e:
                logging.error(f"Error generating from Gemini API: {e}")
                return {"choices": [{"message": {"content": ""}} for _ in range(n)]}
                
        # Return empty response if all retries failed
        return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

async def generate_from_gemini_api(
    full_contexts: List[List[Dict[str, str]]],
    model_name: str = "gemini-2.5-pro-exp-03-25",
    temperature: float = 0.5,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    n: int = 1,
    thinking_budget: int = -1,
    requests_per_minute: int = 100,
    show_progress: bool = True,
    max_concurrent: int = 100,
) -> List[Dict[str, Any]]:
    """Generate from Google Gemini API.

    Args:
        full_contexts: List of message lists to generate from (OpenAI format: [{"role": "user", "content": "..."}]).
        model_name: Model name (default: "gemini-2.5-pro-exp-03-25").
        temperature: Temperature for generation (0-1).
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p sampling parameter.
        n: Number of responses to generate for each API call.
        requests_per_minute: Number of requests per minute to allow.
        show_progress: Whether to show progress bar.
        max_concurrent: Maximum number of concurrent requests.

    Returns:
        List of generated responses in OpenAI-like format.
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_task(messages):
        # Process OpenAI-style messages to Gemini format
        contents = []
        system_prompt = None
        
        # Check for system message first (should be the first message if present)
        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]  # Remove system message
        
        # Convert remaining messages
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Map OpenAI roles to Gemini roles
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                # Skip unknown roles
                continue
                
            # Create Gemini Content object
            gemini_content = types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content)]
            )
            contents.append(gemini_content)
        
        # Only allow max_concurrent tasks to run concurrently
        async with semaphore:
            return await _throttled_gemini_generate_content_acreate(
                client=client,
                model=model_name,
                contents=contents,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                thinking_budget=thinking_budget,
                system_prompt=system_prompt,
                limiter=limiter,
            )

    # Create a task for each context
    async_responses = [limited_task(context) for context in full_contexts]
    
    if show_progress:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    
    return responses

def extract_json(text):
    # Find all potential JSON objects
    match = re.findall(r'\{[^{}]*\}', text)
    if not match:
        return None
    
    last_json_string = match[-1]
    
    # Try to parse the extracted string as JSON
    try:
        return json.loads(last_json_string)
    except json.JSONDecodeError:
        return None
    
def extract_nested_json(text):
    # Try to find a JSON object that starts at the beginning of a line
    matches = re.finditer(r'(?m)^{.*}$', text, re.DOTALL)
    
    # Get the last match
    json_str = None
    for match in matches:
        json_str = match.group()
    
    if not json_str:
        return None
        
    # Try to parse the extracted string as JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If parsing fails, try to clean up the string
        try:
            # Remove any leading/trailing whitespace and newlines
            json_str = json_str.strip()
            # Handle escaped newlines
            json_str = json_str.replace('\\n', '\n')
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None 

async def generate_responses_in_batch(
    full_contexts: List[List[Dict[str, str]]],
    model_name: str,
    temperature: float,
    max_tokens: int,
    n: int = 1,
    thinking_budget: int = -1,
    show_progress: bool = True,
    reasoning_effort: str = "medium",
) -> Union[List[str], List[List[str]]]:

    if model_name in ["claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219", 
                       "claude-sonnet-4-20250514", "claude-sonnet-4-20250514-thinking",
                       "claude-opus-4-1-20250805", "claude-opus-4-1-20250805-thinking",
                       "claude-3-7-sonnet-20250219-thinking"]:
        # Anthropic-like responses (dictionary-based)
        responses = await generate_from_anthropic_chat_completion(
            full_contexts, model_name, temperature=temperature, max_tokens=max_tokens, n=n, show_progress=show_progress
        )
        generated_responses = []
        for resp in responses:
            scenario_responses = []
            for i in range(n):
                # Safely get the choice for index i
                choice = resp.get('choices', [])
                if i < len(choice):
                    content = choice[i].get('message', {}).get('content', "")
                else:
                    content = ""
                try:
                    content = content.strip()
                except:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)

    elif model_name in ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o-241120", "gpt-4.1-2025-04-14", "o3-2025-04-16"]:
        # GPT-like responses (object-based)
        responses = await generate_from_azure_openai_chat_completion(
            full_contexts, model_name, temperature=temperature, max_tokens=max_tokens, n=n, show_progress=show_progress,
            reasoning_effort=reasoning_effort,
        )
        generated_responses = []
        for resp in responses:
            scenario_responses = []
            for i in range(n):
                try:
                    content = resp.choices[i].message.content
                    content = content.strip()
                except:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)

    elif model_name in ["gemini-2.5-pro", "gemini-2.5-flash",
                    "gemini-2.5-pro-exp-03-25", "gemini-2.0-flash", "gemini-2.5-flash-preview-04-17"]:
        # Gemini-like responses (dictionary-based)
        responses = await generate_from_gemini_api(
            full_contexts, model_name, temperature=temperature, max_tokens=max_tokens, n=n, thinking_budget=thinking_budget, show_progress=show_progress
        )

        # return responses
        
        # TODO: Change Back
        generated_responses = []
        for resp in responses:
            scenario_responses = []
            for i in range(n):
                choice = resp.get('choices', [])
                if i < len(choice):
                    content = choice[i].get('message', {}).get('content', "")
                else:
                    content = ""
                try:
                    content = content.strip()
                except:
                    content = ""
                scenario_responses.append(content)
            generated_responses.append(scenario_responses)
    else:
        # Handle other models or raise an exception
        raise ValueError(f"Unsupported model: {model_name}")

    # If n == 1, flatten the responses to a simple list of strings
    if n == 1:
        return [r[0] if r else "" for r in generated_responses]
    else:
        return generated_responses
