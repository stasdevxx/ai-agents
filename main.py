import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class WeatherResult:
    """Data class representing weather information for a city."""
    city: str
    condition: str
    temperature: int
    
    def to_string(self) -> str:
        """Convert weather result to human-readable string."""
        return f"In {self.city}, it is {self.condition} around {self.temperature}°C."


class WeatherService:
    """Service class for weather-related operations."""
    
    WEATHER_CONDITIONS = ["sunny", "rainy", "cloudy", "windy", "stormy", "snowy"]
    TEMP_RANGE = (-5, 35)
    
    @classmethod
    def get_weather(cls, city: str) -> WeatherResult:
        """
        Generate fake weather data for the specified city.
        
        Args:
            city: Name of the city to get weather for
            
        Returns:
            WeatherResult object containing weather information
        """
        condition = random.choice(cls.WEATHER_CONDITIONS)
        temperature = random.randint(*cls.TEMP_RANGE)
        
        return WeatherResult(
            city=city,
            condition=condition,
            temperature=temperature
        )


class OpenAIAgent:
    """
    Professional AI Agent implementation using OpenAI's ChatCompletion API.
    
    This agent demonstrates tool calling functionality by using a weather
    lookup tool to respond to user queries about weather conditions.
    """
    
    MODEL = "gpt-4o-mini"
    DEFAULT_CITY = "Prague"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI agent.
        
        Args:
            api_key: OpenAI API key. If None, will be loaded from environment.
        """
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure and return a logger instance."""
        logger = logging.getLogger(self.__class__.__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Define the available tools for the agent.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        return [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a specified city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city to get weather for (e.g., 'Prague')"
                        }
                    },
                    "required": ["city"],
                    "additionalProperties": False
                }
            }
        }]
    
    def _create_initial_messages(self, user_prompt: str) -> List[Dict[str, Any]]:
        """
        Create the initial message sequence for the conversation.
        
        Args:
            user_prompt: The user's input query
            
        Returns:
            List of messages in OpenAI format
        """
        system_message = (
            "You are a helpful weather assistant. When a user asks about weather, "
            "extract the city name from their message and use the get_weather tool "
            "to provide accurate weather information. Always call the tool exactly once "
            "and provide a natural, helpful response based on the results."
        )
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    
    def _execute_tool_call(self, tool_call) -> str:
        """
        Execute a single tool call and return the result.
        
        Args:
            tool_call: OpenAI tool call object
            
        Returns:
            String result from tool execution
        """
        if tool_call.type != "function" or tool_call.function.name != "get_weather":
            raise ValueError(f"Unknown tool: {tool_call.function.name}")
        
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
            city = arguments.get("city", self.DEFAULT_CITY)
            
            self.logger.info(f"Executing weather lookup for city: {city}")
            weather_result = WeatherService.get_weather(city)
            
            return weather_result.to_string()
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse tool arguments: {e}")
            # Fallback to default city
            weather_result = WeatherService.get_weather(self.DEFAULT_CITY)
            return weather_result.to_string()
    
    def process_query(self, user_prompt: str) -> str:
        """
        Process a user query using the OpenAI agent with tool calling.
        
        This method implements the standard tool calling flow:
        1. Send user query with available tools
        2. Execute any requested tool calls
        3. Send tool results back to get final response
        
        Args:
            user_prompt: The user's input query
            
        Returns:
            Final response string from the agent
        """
        try:
            self.logger.info(f"Processing user query: {user_prompt}")
            
            messages = self._create_initial_messages(user_prompt)
            tools = self._get_tool_definitions()
            
            # First API call - let the model decide on tool usage
            self.logger.info("Making initial API call to OpenAI")
            first_response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"  # Let the model decide when to use tools
            )
            
            assistant_message = first_response.choices[0].message
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": assistant_message.tool_calls
            })
            
            # Execute tool calls if any were made
            if assistant_message.tool_calls:
                self.logger.info(f"Executing {len(assistant_message.tool_calls)} tool call(s)")
                
                for tool_call in assistant_message.tool_calls:
                    tool_result = self._execute_tool_call(tool_call)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result
                    })
                
                # Second API call - get final response with tool results
                self.logger.info("Making final API call with tool results")
                final_response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=messages
                )
                
                result = final_response.choices[0].message.content or ""
                
            else:
                # No tools were called, use the initial response
                result = assistant_message.content or ""
            
            self.logger.info("Query processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"


def main():
    """Main entry point for the professional weather agent."""
    
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:]).strip()
    else:
        user_query = "What's the weather in Prague?"
    
    # Initialize and run the agent
    try:
        agent = OpenAIAgent()
        response = agent.process_query(user_query)
        print(response)
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()