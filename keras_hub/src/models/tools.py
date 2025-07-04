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

import dataclasses

@dataclasses.dataclass
class Tool:
    """A class to represent a tool that a CausalLM model can use.

    Attributes:
        name: The name of the tool.
        description: A description of what the tool does.
        parameters: A JSON schema object describing the parameters of the tool.
    """
    name: str
    description: str | None = None
    parameters: dict | None = None
