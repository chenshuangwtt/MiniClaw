"""MiniClaw demo — run an agent with FakeLLM and a couple of tools.

Usage:
    python examples/demo.py
"""

from miniclaw.agent_loop import Agent
from miniclaw.llm.fake import FakeLLM
from miniclaw.tool_registry import ToolRegistry
from miniclaw.trace import TraceLogger

# --- Define tools ---
tools = ToolRegistry()


@tools.register(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)
def get_weather(city: str) -> str:
    # Fake weather data
    data = {
        "Beijing": "Sunny, 25°C",
        "Shanghai": "Cloudy, 22°C",
        "Tokyo": "Rainy, 18°C",
    }
    return data.get(city, f"No data for {city}")


@tools.register(
    name="calculate",
    description="Evaluate a math expression.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression, e.g. '2+3*4'"},
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --- Script the LLM responses ---
llm = FakeLLM([
    # Turn 1: call get_weather
    '{"tool_call": {"name": "get_weather", "arguments": {"city": "Beijing"}}}',
    # Turn 2: call calculate
    '{"tool_call": {"name": "calculate", "arguments": {"expression": "25 * 9/5 + 32"}}}',
    # Turn 3: final answer
    "The weather in Beijing is Sunny, 25°C (that's 77°F).",
])

# --- Build and run ---
trace = TraceLogger(console=True)
agent = Agent(
    llm=llm,
    tools=tools,
    trace=trace,
    max_turns=5,
)

print("🐾 MiniClaw Demo\n")
reply = agent.run("What's the weather in Beijing? Convert the temperature to Fahrenheit.")
print(f"\n🤖 Agent: {reply}")
print(f"\n📊 Trace events: {len(trace.get_events())}")
