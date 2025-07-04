# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Title: Function Calling with GemmaCausalLM
Author: [Your Name]
Date created: YYYY/MM/DD
Last modified: YYYY/MM/DD
Description: Example of how to use function calling with GemmaCausalLM.
"""

import keras_hub
from keras_hub.src.models.tools import Tool

# Define a simple tool
get_weather_tool = Tool(
    name="get_weather",
    description="Gets the current weather in a given city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city to get the weather for.",
            }
        },
        "required": ["city"],
    },
)

# Instantiate GemmaCausalLM with the tool
# Note: You'll need to have a Gemma preset downloaded or available.
# For this example, we assume a preset like "gemma_2b_en" is available.
# Replace with a valid preset if needed.
try:
    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
        "gemma_2b_en", tools=[get_weather_tool]
    )
except Exception as e:
    print(f"Error loading Gemma preset. Please ensure a valid preset is available: {e}")
    print("Skipping function calling example.")
    exit()


gemma_lm.compile(sampler="greedy")

# Craft a prompt that should trigger the tool
prompt = "What's the weather like in London?"

print(f"Prompt: {prompt}")

# Generate with the prompt
output = gemma_lm.generate(prompt, max_length=64)

print(f"Output: {output}")

# Expected output (will vary depending on the model's generation):
# Output: {'name': 'get_weather', 'arguments': {'city': 'London'}}
# Or it might be a string containing the JSON if parsing is not perfect.
# Or it might be a text response if the model doesn't choose to call the function.

# Example of how to handle the output
if isinstance(output, dict) and output.get("name") == "get_weather":
    city = output["arguments"].get("city")
    print(f"\nModel wants to call 'get_weather' for city: {city}")
    # In a real application, you would now call your actual get_weather function
    # and potentially feed the result back to the model.
    # For example:
    # weather_result = your_get_weather_function(city)
    # response = gemma_lm.generate(f"The weather in {city} is {weather_result}. What should I wear?")
    # print(f"Final response: {response}")
else:
    print("\nModel did not generate a function call for 'get_weather'.")
    print(f"Generated text: {output}")
