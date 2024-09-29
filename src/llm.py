from dotenv import load_dotenv
from openai import OpenAI
from openai.types import CompletionUsage
from pydantic import BaseModel
from typing import Type, TypeVar, Optional, Union, overload
import logging
# For local debugging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()
client = OpenAI()

def estimate_llm_cost(model: str, usage: CompletionUsage) -> float:
    """
    Estimates the cost of querying the LLM based on the number of tokens in the prompt and completion.

    Args:
        completion (Completion): The completion object returned by the LLM.

    Returns:
        float: The estimated cost of querying the LLM.
    """

    # Define the pricing for each model
    pricing = {
        "gpt-4o-mini-2024-07-18": {"prompt_cost_per_million_tokens": 0.150, "completion_cost_per_million_tokens": 0.60},
    }

    if model not in pricing:
        logging.error(f"Model '{model}' is not supported for cost estimation.")
        return 0.0

    prompt_cost_per_million_tokens = pricing[model]["prompt_cost_per_million_tokens"]
    completion_cost_per_million_tokens = pricing[model]["completion_cost_per_million_tokens"]

    prompt_cost = (usage.prompt_tokens / 1_000_000) * prompt_cost_per_million_tokens
    completion_cost = (usage.completion_tokens / 1_000_000) * completion_cost_per_million_tokens

    return prompt_cost + completion_cost

T = TypeVar('T', bound=BaseModel)
@overload
def query_llm(prompt: str, response_format: Type[T]) -> T:
    ...
@overload
def query_llm(prompt: str, response_format: None = None) -> str:
    ...
def query_llm(prompt: str, response_format: Optional[Type[T]] = None) -> Optional[Union[T, str]]:
    """
    Queries an LLM with a given prompt and returns the response.

    Args:
        prompt (str): The prompt to send to the LLM.
        response_format (Optional[Type[BaseModel]]): The Pydantic model to parse the response.

    Returns:
        Optional[Union[T, str]]: An instance of the response_format model containing the LLM's response,
        or a string if no response_format is provided, or None if an error occurs.
    """
    logging.info(f"Querying LLM with prompt: {prompt}")
    try:
        # If a response_format is provided, use it to parse the response
        if response_format:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format
            )
            logging.info(f"Successfully queried LLM!")
            if completion.usage:
                estimated_cost = estimate_llm_cost(completion.model, completion.usage)
                logging.info(f"Estimated cost of querying LLM: ${estimated_cost:.4f}")
            return completion.choices[0].message.parsed if completion.choices else None
        else:
            # If no response_format, return the content as a string
            completion = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": prompt}],
            )
            logging.info(f"Successfully queried LLM!")
            if completion.usage:
                estimated_cost = estimate_llm_cost(completion.model, completion.usage)
                logging.info(f"Estimated cost of querying LLM: ${estimated_cost:.4f}")
            return completion.choices[0].message.content if completion.choices else None

    except Exception as e:
        print(f"An error occurred while querying LLM: {e}")
        return None

# sanity check logging + LLM key is working
if __name__ == "__main__":
    prompt = "Write a poem about the sea."
    response = query_llm(prompt)
    print("Response from LLM:", response)
