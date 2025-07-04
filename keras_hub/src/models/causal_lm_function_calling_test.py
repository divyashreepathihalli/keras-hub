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

import pytest
import keras
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.tools import Tool
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.tokenizers.tokenizer import Tokenizer


class MockTokenizer(Tokenizer):
    def vocabulary_size(self):
        return 10

    def tokenize(self, inputs):
        return inputs  # Dummy implementation

    def detokenize(self, inputs):
        # If it's a dict (from generate_postprocess), handle it.
        if isinstance(inputs, dict) and "token_ids" in inputs:
            return inputs["token_ids"] # or some string representation
        return inputs # Dummy implementation

    @property
    def end_token_id(self):
        return 0


class MockPreprocessor(Preprocessor):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def generate_preprocess(self, inputs, sequence_length=None):
        # Simulate preprocessing that returns a dict
        return {"token_ids": inputs, "padding_mask": keras.ops.ones_like(inputs, dtype="bool")}

    def generate_postprocess(self, inputs):
        # In a real scenario, this would detokenize token IDs.
        # For testing function calling, we expect `inputs` to be the raw model output string (or dict of tokens).
        # The CausalLM.generate() method's postprocess will try to parse this string as JSON.
        if isinstance(inputs, dict) and "token_ids" in inputs:
            return inputs["token_ids"] # This would be the string output from the LLM
        return inputs


class MockCausalLM(CausalLM):
    def __init__(self, backbone=None, preprocessor=None, mock_output_text=None, tools=None, *args, **kwargs):
        # First, set up the attributes that Task expects to be present,
        # like self.backbone and self.preprocessor.
        if backbone is None:
            # Provide a minimal backbone mock.
            backbone = keras.Sequential([keras.Input(shape=(1,)), keras.layers.Dense(10)], name="mock_backbone")

        # These need to be set *before* CausalLM's (and therefore Task's) __init__ is called.
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.mock_output_text = mock_output_text

        # Now call super().__init__ for CausalLM.
        # CausalLM.__init__ takes `tools` and `**kwargs` (which go to Task, then Model, then Layer).
        # We must ensure `backbone` and `preprocessor` are not in `kwargs` passed to CausalLM's super call
        # as they are not standard Layer/Model kwargs.
        # The `tools` kwarg is handled by CausalLM's __init__.
        super().__init__(*args, tools=tools, **kwargs) # `compile` is a kwarg for Task, will be in kwargs if passed.


    def generate_step(self, inputs, stop_token_ids=None):
        # Return a structure that generate_postprocess in CausalLM can handle
        # This should be the "raw" output of the model before JSON parsing.
        # So, it should be what preprocessor.generate_postprocess would return.
        # In our test, this means it should be the mock_output_text itself.
        return {"token_ids": self.mock_output_text, "padding_mask": keras.ops.ones_like(inputs["token_ids"], dtype="bool")}

    def make_generate_function(self):
        # Override to prevent actual compilation, just return the step function
        return self.generate_step


@pytest.fixture
def get_weather_tool():
    return Tool(
        name="get_weather",
        description="Gets the current weather in a given city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city to get the weather for."}},
            "required": ["city"],
        },
    )

def test_function_call_parsing(get_weather_tool):
    mock_output = 'Some text then {"name": "get_weather", "arguments": {"city": "London"}} and some more text'
    mock_tokenizer = MockTokenizer()
    mock_preprocessor = MockPreprocessor(tokenizer=mock_tokenizer)

    model = MockCausalLM(tools=[get_weather_tool], preprocessor=mock_preprocessor, mock_output_text=mock_output)
    model.compile(sampler="greedy") # Need to compile to set sampler

    output = model.generate("What's the weather in London?")
    assert isinstance(output, dict)
    assert output["name"] == "get_weather"
    assert output["arguments"] == {"city": "London"}

def test_no_function_call(get_weather_tool):
    mock_output = "The weather is nice today."
    mock_tokenizer = MockTokenizer()
    mock_preprocessor = MockPreprocessor(tokenizer=mock_tokenizer)

    model = MockCausalLM(tools=[get_weather_tool], preprocessor=mock_preprocessor, mock_output_text=mock_output)
    model.compile(sampler="greedy")

    output = model.generate("Tell me a joke.")
    assert isinstance(output, str)
    assert output == "The weather is nice today."

def test_malformed_json_returns_string(get_weather_tool):
    mock_output = '{"name": "get_weather", "arguments": {"city": "London"}' # Malformed JSON
    mock_tokenizer = MockTokenizer()
    mock_preprocessor = MockPreprocessor(tokenizer=mock_tokenizer)

    model = MockCausalLM(tools=[get_weather_tool], preprocessor=mock_preprocessor, mock_output_text=mock_output)
    model.compile(sampler="greedy")

    output = model.generate("What's the weather in London?")
    assert isinstance(output, str)
    assert mock_output in output # The original string should be part of the output

def test_json_not_a_tool_call_returns_string(get_weather_tool):
    mock_output = '{"info": "some_data", "value": 123}' # Valid JSON but not a tool call
    mock_tokenizer = MockTokenizer()
    mock_preprocessor = MockPreprocessor(tokenizer=mock_tokenizer)

    model = MockCausalLM(tools=[get_weather_tool], preprocessor=mock_preprocessor, mock_output_text=mock_output)
    model.compile(sampler="greedy")

    output = model.generate("What's the weather in London?")
    assert isinstance(output, str)
    assert mock_output in output

def test_function_call_with_batched_input(get_weather_tool):
    mock_output_1 = 'Some text then {"name": "get_weather", "arguments": {"city": "London"}}'
    mock_output_2 = 'Some text then {"name": "get_weather", "arguments": {"city": "Paris"}}'

    # For batched inputs, generate_step would be called per batch.
    # We need to simulate that the MockCausalLM's mock_output_text can vary or is a list.
    # Let's adjust MockCausalLM or how we use it for this test.
    # For simplicity, let's assume generate() processes a list of prompts and our mock_output_text
    # is used for each. This isn't ideal but tests the parsing logic in postprocess.
    # A better mock would handle batched inputs in generate_step.

    # The current CausalLM.generate() processes inputs one by one if they are in a list.
    # The postprocess function handles a list of generated texts.

    mock_tokenizer = MockTokenizer()
    mock_preprocessor = MockPreprocessor(tokenizer=mock_tokenizer)

    # This mock will return the same text for all items in a batch.
    # To test batching properly, generate_step would need to be more sophisticated
    # or we'd need to mock the output of generate_function more directly.
    # However, the JSON parsing in CausalLM's postprocess is applied element-wise if it gets a list.

    model_london = MockCausalLM(tools=[get_weather_tool], preprocessor=mock_preprocessor, mock_output_text=mock_output_1)
    model_london.compile(sampler="greedy")
    output1 = model_london.generate("Weather in London?")


    model_paris = MockCausalLM(tools=[get_weather_tool], preprocessor=mock_preprocessor, mock_output_text=mock_output_2)
    model_paris.compile(sampler="greedy")
    output2 = model_paris.generate("Weather in Paris?")

    outputs = [output1, output2] # Manually simulate batch output collection for this test structure

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    assert isinstance(outputs[0], dict)
    assert outputs[0]["name"] == "get_weather"
    assert outputs[0]["arguments"] == {"city": "London"}
    assert isinstance(outputs[1], dict)
    assert outputs[1]["name"] == "get_weather"
    assert outputs[1]["arguments"] == {"city": "Paris"}

def test_tool_prompt_injection():
    tool = Tool(name="my_tool", description="Does something cool.")
    mock_tokenizer = MockTokenizer()
    mock_preprocessor = MockPreprocessor(tokenizer=mock_tokenizer)

    original_prompt = "Hello world"

    def capture_preprocessor_input(inputs, sequence_length=None):
        capture_preprocessor_input.called_with = inputs
        return {"token_ids": inputs, "padding_mask": keras.ops.ones_like(inputs, dtype="bool")}
    capture_preprocessor_input.called_with = None

    mock_preprocessor.generate_preprocess = capture_preprocessor_input

    model = MockCausalLM(tools=[tool], preprocessor=mock_preprocessor, mock_output_text="Some output")
    model.compile(sampler="greedy")
    model.generate(original_prompt)

    assert capture_preprocessor_input.called_with is not None
    # generate_preprocess is called with a list of strings if the input is a list of strings
    # or a single string if input is a single string.
    # _normalize_generate_inputs wraps single string in a list, so preprocess in generate() gets a list.
    # And the preprocess in CausalLM.generate() also iterates:
    # `inputs = [preprocess(x) for x in inputs]`
    # `x` here is `tool_prompt + "\n\nUser query: " + i` which is a string.
    # So capture_preprocessor_input.called_with should be the string itself.
    # Let's re-check the CausalLM.generate() flow for preprocess.
    # inputs_iterable, input_is_scalar = self._normalize_generate_inputs(inputs) -> inputs_iterable is list of batches
    #   if self.preprocessor is not None:
    #       for x_batch in inputs_iterable:  <- x_batch is the actual input to preprocess() defined in generate()
    #           processed_inputs.append(preprocess(x_batch))
    # The preprocess() defined in generate() is:
    #   def preprocess(x):
    #       if self.tools: x = tool_prompt + ... + x # x is string here
    #       return self.preprocessor.generate_preprocess(x, sequence_length=max_length)
    # So, self.preprocessor.generate_preprocess (which is capture_preprocessor_input) is called with the modified string
    # that has been wrapped in a list by CausalLM.generate's preprocessing logic.

    processed_prompt_list = capture_preprocessor_input.called_with
    assert isinstance(processed_prompt_list, list), f"Expected list, got {type(processed_prompt_list)}"
    assert len(processed_prompt_list) == 1, f"Expected list of length 1, got {len(processed_prompt_list)}"
    processed_prompt_str = processed_prompt_list[0]
    assert isinstance(processed_prompt_str, str), f"Expected string element, got {type(processed_prompt_str)}"

    assert "You have access to the following tools:" in processed_prompt_str
    assert "- my_tool: Does something cool." in processed_prompt_str
    assert "User query: Hello world" in processed_prompt_str
