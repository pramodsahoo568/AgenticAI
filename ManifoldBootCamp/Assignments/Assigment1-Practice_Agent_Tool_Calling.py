from langchain.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

load_dotenv()
import json

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city
    """
    # Mock implementation
    weather_data = {
        "bangalore": "Sunny, 28°C",
        "mumbai": "Rainy, 26°C",
        "delhi": "Cloudy, 22°C"
    }
    return weather_data.get(city.lower(), "Weather data not available")


@tool
def book_flight(origin: str, destination: str, date: str) -> dict:
    """Book a flight from one city to another.

    Args:
        origin: The origin city
        destination: The destination city
        date: The date of the flight
    """
    # Mock implementation
    return {
        "booking_id": "1234567890",
        "route": f"{origin} to {destination}",
        "date": date,
        "status": "confirmed"
    }

# example tool
@tool
def best_food(city: str) -> str:
    """Get the best  food in a city.
        Args:   city: The name of the city """
    # Mock implementation
    best_food_data = {
        "bangalore": "Masala Dosa",
        "mumbai": "Vada Pav",    "delhi": "Chaat"
    }
    return best_food_data.get(city.lower(), "Best food data not available")


# list of tools
tools = [get_weather, book_flight, best_food]
tool_names = {t.name: t for t in tools}
print("Tool Names: ", tool_names)
print("*************************************************")

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=1).bind_tools(tools)
print("*************************************************")

def run(prompt: str):
    print("*************************************************")
    print("Prompt: ", prompt)
    msg = llm_openai.invoke([HumanMessage(content=prompt)])
    print("Model Output: ", msg.content)
    print("Model Tools Calls:", msg.tool_calls)
    print("*************************************************")
    print("Model Full output: ", msg)
    print("*************************************************")

    messages = [HumanMessage(content=prompt), msg]  # combine the prompt and the model output
    for call in msg.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        tool_id = call["id"]
        tool_result = tool_names[tool_name].invoke(tool_args)
    print("Tool Result: ", tool_result)
    print("*************************************************")
    messages.append(ToolMessage(content=json.dumps(tool_result), tool_call_id=tool_id))
    print("Messages: ", messages)
    print("*************************************************")
    final_msg = llm_openai.invoke(messages)
    print("Final Model Output: ", final_msg.content)


#run("What is the weather in Bangalore?")
#run("Book a flight from Bangalore to Mumbai on 12/01/2026")

run("What is the best food in Bangalore")

