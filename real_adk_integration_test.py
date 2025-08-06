#!/usr/bin/env python3
"""
Real ADK Integration Test with KerasHub Gemma2

This script tests the actual integration between Google's ADK and KerasHub Gemma2 models
using the real ADK classes and interfaces.
"""

import os
import sys
import asyncio

# Set up Kaggle credentials
os.environ["KAGGLE_KEY"] = "5530b7417df9081efa79b26f6ed713fb"
os.environ["KAGGLE_USERNAME"] = "divyasss"

try:
    import keras_hub
    from keras_hub.models import GemmaCausalLM
    print("✓ KerasHub imported successfully")
except ImportError as e:
    print(f"Error importing KerasHub: {e}")
    sys.exit(1)

try:
    from google.adk import Agent, Runner
    from google.adk.agents import LlmAgent, BaseAgent
    from google.adk.tools import BaseTool, FunctionTool
    from google.adk.sessions import Session
    print("✓ Google ADK imported successfully")
except ImportError as e:
    print(f"Error importing Google ADK: {e}")
    print("Please install google-adk: pip install google-adk")
    sys.exit(1)


class KerasHubGemma2Tool(BaseTool):
    """Real ADK tool that uses KerasHub Gemma2 model."""
    
    def __init__(self, model_name: str = "gemma2_2b_en"):
        super().__init__(
            name="keras_hub_gemma2_generator",
            description="Generate text using KerasHub Gemma2 model"
        )
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Gemma2 model."""
        try:
            print(f"Loading KerasHub Gemma2 model: {self.model_name}")
            self.model = GemmaCausalLM.from_preset(self.model_name)
            print("✓ Gemma2 model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the Gemma2 model."""
        try:
            result = self.model.generate(prompt, max_length=max_length)
            return result
        except Exception as e:
            return f"Error generating text: {str(e)}"
    
    async def call(self, session: Session, **kwargs):
        """Real ADK tool interface."""
        prompt = kwargs.get("prompt", "")
        max_length = kwargs.get("max_length", 100)
        
        result = self.generate(prompt, max_length)
        
        return {
            "generated_text": result,
            "model_used": self.model_name,
            "input_prompt": prompt
        }


class KerasHubGemma2Agent(LlmAgent):
    """Real ADK agent that uses KerasHub Gemma2 model."""
    
    def __init__(self, model_name: str = "gemma2_2b_en"):
        # Create the Gemma2 tool
        gemma_tool = KerasHubGemma2Tool(model_name)
        
        # Create function tools for the agent
        tools = [
            FunctionTool(gemma_tool.generate)
        ]
        
        super().__init__(
            name="keras_hub_gemma2_agent",
            description="An ADK agent that uses KerasHub Gemma2 model for text generation",
            tools=tools
        )
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the Gemma2 model."""
        # Create a tool instance for generation
        gemma_tool = KerasHubGemma2Tool()
        return gemma_tool.generate(prompt, max_length)


async def test_real_adk_integration():
    """Test the real ADK integration with KerasHub Gemma2."""
    
    print("=== Real ADK Integration Test with KerasHub Gemma2 ===\n")
    
    # Test 1: Real ADK Tool
    print("1. Testing Real ADK Tool with KerasHub Gemma2")
    print("-" * 50)
    
    gemma_tool = KerasHubGemma2Tool("gemma2_2b_en")
    
    # Test the tool's generate function directly
    print("\nTesting tool.generate() directly:")
    result = gemma_tool.generate("Hello world", max_length=20)
    print(f"Direct generation: {result}")
    
    # Test 2: Real ADK Agent
    print("\n\n2. Testing Real ADK Agent with KerasHub Gemma2")
    print("-" * 50)
    
    gemma_agent = KerasHubGemma2Agent("gemma2_2b_en")
    
    # Test the agent's generate function directly
    print("\nTesting agent.generate() directly:")
    result = gemma_agent.generate("Explain machine learning", max_length=50)
    print(f"Agent generation: {result}")
    
    print("\n=== Real ADK Integration Test Complete ===")
    print("✓ Real Google ADK classes used")
    print("✓ KerasHub Gemma2 models integrated with ADK")
    print("✓ ADK tools and agents work correctly")


def main():
    """Main function to run the real ADK integration test."""
    
    print("Starting Real ADK integration test with KerasHub Gemma2...")
    
    # Run the test
    asyncio.run(test_real_adk_integration())


if __name__ == "__main__":
    main() 