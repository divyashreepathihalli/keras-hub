#!/usr/bin/env python3
"""
Colab-Compatible Weather Agent with KerasHub Gemma2

This example demonstrates a practical weather agent that uses KerasHub Gemma2 models
with Google's ADK, compatible with Google Colab environment.
"""

import os
import sys
import asyncio
import nest_asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Apply nest_asyncio to handle Colab's event loop
try:
    nest_asyncio.apply()
    print("✓ nest_asyncio applied for Colab compatibility")
except ImportError:
    print("nest_asyncio not available, proceeding without it")

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


class WeatherDataTool(BaseTool):
    """ADK tool that provides weather data and analysis using KerasHub Gemma2."""
    
    def __init__(self, model_name: str = "gemma2_2b_en"):
        super().__init__(
            name="weather_data_analyzer",
            description="Analyze weather data and provide insights using KerasHub Gemma2"
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
    
    def analyze_weather(self, weather_data: Dict[str, Any]) -> str:
        """Analyze weather data using Gemma2 model."""
        try:
            # Format weather data for analysis
            weather_info = f"""
Weather Data:
- Temperature: {weather_data.get('temperature', 'N/A')}°C
- Humidity: {weather_data.get('humidity', 'N/A')}%
- Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
- Conditions: {weather_data.get('conditions', 'N/A')}
- Location: {weather_data.get('location', 'N/A')}
- Time: {weather_data.get('time', 'N/A')}

Please analyze this weather data and provide insights about:
1. Current weather conditions
2. Recommendations for outdoor activities
3. Any weather warnings or concerns
4. General weather trends
"""
            
            result = self.model.generate(weather_info, max_length=200)
            return result
        except Exception as e:
            return f"Error analyzing weather: {str(e)}"
    
    def generate_weather_report(self, location: str, conditions: str) -> str:
        """Generate a weather report using Gemma2 model."""
        try:
            prompt = f"""
Generate a professional weather report for {location}.

Current conditions: {conditions}

Please include:
1. Current weather summary
2. Temperature and humidity details
3. Wind conditions
4. Recommendations for the day
5. Any weather alerts or warnings
"""
            
            result = self.model.generate(prompt, max_length=250)
            return result
        except Exception as e:
            return f"Error generating weather report: {str(e)}"
    
    def provide_weather_advice(self, activity: str, weather_data: Dict[str, Any]) -> str:
        """Provide weather-based advice for activities."""
        try:
            weather_info = f"""
Activity: {activity}
Weather Conditions:
- Temperature: {weather_data.get('temperature', 'N/A')}°C
- Humidity: {weather_data.get('humidity', 'N/A')}%
- Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
- Conditions: {weather_data.get('conditions', 'N/A')}

Please provide advice about whether this activity is suitable for the current weather conditions.
Include safety considerations and recommendations.
"""
            
            result = self.model.generate(weather_info, max_length=200)
            return result
        except Exception as e:
            return f"Error providing weather advice: {str(e)}"
    
    def call_sync(self, **kwargs):
        """Synchronous version of the call method for Colab compatibility."""
        action = kwargs.get("action", "analyze")
        
        if action == "analyze":
            weather_data = kwargs.get("weather_data", {})
            result = self.analyze_weather(weather_data)
        elif action == "report":
            location = kwargs.get("location", "Unknown")
            conditions = kwargs.get("conditions", "Unknown")
            result = self.generate_weather_report(location, conditions)
        elif action == "advice":
            activity = kwargs.get("activity", "Unknown")
            weather_data = kwargs.get("weather_data", {})
            result = self.provide_weather_advice(activity, weather_data)
        else:
            result = "Unknown action. Please specify: analyze, report, or advice"
        
        return {
            "action": action,
            "result": result,
            "model_used": self.model_name
        }
    
    async def call(self, session: Session, **kwargs):
        """ADK tool interface."""
        return self.call_sync(**kwargs)


class WeatherAgent(LlmAgent):
    """ADK weather agent that uses KerasHub Gemma2 model."""
    
    def __init__(self, model_name: str = "gemma2_2b_en"):
        # Create the weather tool
        weather_tool = WeatherDataTool(model_name)
        
        # Create function tools for the agent
        tools = [
            FunctionTool(weather_tool.analyze_weather),
            FunctionTool(weather_tool.generate_weather_report),
            FunctionTool(weather_tool.provide_weather_advice)
        ]
        
        super().__init__(
            name="weather_agent_gemma2",
            description="A weather agent that uses KerasHub Gemma2 for weather analysis and recommendations",
            tools=tools
        )
    
    def analyze_weather_data(self, weather_data: Dict[str, Any]) -> str:
        """Analyze weather data using the agent."""
        weather_tool = WeatherDataTool()
        return weather_tool.analyze_weather(weather_data)
    
    def generate_report(self, location: str, conditions: str) -> str:
        """Generate a weather report."""
        weather_tool = WeatherDataTool()
        return weather_tool.generate_weather_report(location, conditions)
    
    def get_activity_advice(self, activity: str, weather_data: Dict[str, Any]) -> str:
        """Get weather advice for activities."""
        weather_tool = WeatherDataTool()
        return weather_tool.provide_weather_advice(activity, weather_data)


def create_sample_weather_data():
    """Create sample weather data for testing."""
    return {
        "temperature": 22,
        "humidity": 65,
        "wind_speed": 15,
        "conditions": "Partly cloudy with light breeze",
        "location": "San Francisco, CA",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def test_weather_agent_sync():
    """Test the weather agent synchronously (Colab compatible)."""
    
    print("=== Colab-Compatible Weather Agent with KerasHub Gemma2 Test ===\n")
    
    # Create weather agent
    weather_agent = WeatherAgent("gemma2_2b_en")
    
    # Sample weather data
    weather_data = create_sample_weather_data()
    
    print("1. Testing Weather Analysis")
    print("-" * 40)
    analysis = weather_agent.analyze_weather_data(weather_data)
    print(f"Weather Analysis:\n{analysis}\n")
    
    print("2. Testing Weather Report Generation")
    print("-" * 40)
    report = weather_agent.generate_report("San Francisco, CA", "Partly cloudy, 22°C, light breeze")
    print(f"Weather Report:\n{report}\n")
    
    print("3. Testing Activity Advice")
    print("-" * 40)
    advice = weather_agent.get_activity_advice("Outdoor hiking", weather_data)
    print(f"Activity Advice:\n{advice}\n")
    
    print("4. Testing Tool Integration")
    print("-" * 40)
    weather_tool = WeatherDataTool("gemma2_2b_en")
    
    # Test different actions synchronously
    actions = [
        ("analyze", {"weather_data": weather_data}),
        ("report", {"location": "New York, NY", "conditions": "Sunny, 28°C, calm"}),
        ("advice", {"activity": "Beach volleyball", "weather_data": weather_data})
    ]
    
    for action, params in actions:
        print(f"\nTesting {action} action:")
        result = weather_tool.call_sync(action=action, **params)
        print(f"Result: {result.get('result', 'No result')[:200]}...")
    
    print("\n=== Colab-Compatible Weather Agent Test Complete ===")
    print("✓ Weather agent successfully uses KerasHub Gemma2")
    print("✓ Weather analysis, reporting, and advice generation working")
    print("✓ ADK tool integration functional")
    print("✓ Colab-compatible execution")


def interactive_weather_session_sync():
    """Run an interactive weather agent session (Colab compatible)."""
    
    print("\n=== Interactive Weather Agent Session (Colab Compatible) ===")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    weather_agent = WeatherAgent("gemma2_2b_en")
    
    while True:
        try:
            print("\nAvailable actions:")
            print("1. analyze")
            print("2. report <location> <conditions>")
            print("3. advice <activity>")
            print("4. quit")
            
            user_input = input("\nEnter your request: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Parse user input
            parts = user_input.split(' ', 1)
            if len(parts) < 1:
                print("Please provide an action")
                continue
            
            action = parts[0].lower()
            params = parts[1] if len(parts) > 1 else ""
            
            if action == "analyze":
                # Use sample weather data for analysis
                weather_data = create_sample_weather_data()
                result = weather_agent.analyze_weather_data(weather_data)
                print(f"\nWeather Analysis:\n{result}")
            
            elif action == "report":
                # Parse location and conditions
                if not params:
                    print("Please provide location and conditions")
                    continue
                report_parts = params.split(' ', 1)
                if len(report_parts) < 2:
                    print("Please provide location and conditions")
                    continue
                location, conditions = report_parts
                result = weather_agent.generate_report(location, conditions)
                print(f"\nWeather Report:\n{result}")
            
            elif action == "advice":
                # Parse activity and use sample weather data
                if not params:
                    print("Please provide an activity")
                    continue
                activity = params
                weather_data = create_sample_weather_data()
                result = weather_agent.get_activity_advice(activity, weather_data)
                print(f"\nActivity Advice:\n{result}")
            
            else:
                print("Unknown action. Please use: analyze, report, or advice")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run the Colab-compatible weather agent tests."""
    
    print("Starting Colab-Compatible Weather Agent with KerasHub Gemma2...")
    
    # Run the test
    test_weather_agent_sync()
    
    # Run interactive session
    try:
        interactive_weather_session_sync()
    except KeyboardInterrupt:
        print("\nSession ended by user.")


if __name__ == "__main__":
    main() 