from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import operator
from dotenv import load_dotenv
load_dotenv()



# create State
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    should_escalate: bool
    issue_type: str
    user_tier: str

# Tools
@tool
def check_order_status(order_id: str) -> dict:
    """Check the status of an order."""
    return {"order_id": order_id, "status": "shipped", "eta": "2024-01-20"}

@tool
def create_ticket(issue: str, priority: str) -> dict:
    """Create a support ticket."""
    return {"ticket_id": "TKT12345", "issue": issue, "priority": priority}

# Setup
tools = [check_order_status, create_ticket]

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_with_tools = llm.bind_tools(tools)

## Add new node
# 2. Check tier node
def check_user_tier_node(state: SupportState):
    """Check user tier (mock implementation)"""
    # In production, look up user in database
    # For now, mock based on message content
    messages = state["messages"]
    first_message = messages[0].content.lower()

    if "vip" in first_message or "premium" in first_message:
        return {"user_tier": "vip"}
    else:
        return {"user_tier": "standard"}

# 3. Routing function
def route_by_tier(state: SupportState) -> str:
    """Route based on user tier"""
    if state.get("user_tier") == "vip":
        return "vip_path"
    return "standard_path"

# 4. VIP-specific node (auto-resolves)
def vip_agent_node(state: SupportState):
    """VIP agent - no escalation"""
    messages = state["messages"]
    # Could use different prompt for VIP
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "should_escalate": False}


# Nodes
def agent_node(state: SupportState):
    ''' Node to handle conversation. and tool calls'''
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Nodes billing_agent_node
def billing_agent_node(state: SupportState):
    ''' Node to handle billing/refund questions'''
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: SupportState) -> Literal["continue", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

# Build graph
workflow = StateGraph(SupportState)
workflow.add_node("check_tier", check_user_tier_node)
workflow.add_node("vip_agent", vip_agent_node)
workflow.add_node("standard_agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("billing_agent", billing_agent_node)

# Set entry point
workflow.set_entry_point("check_tier")

# Route by tier
workflow.add_conditional_edges(
    "check_tier",
    route_by_tier,
    {
        "vip_path": "vip_agent",
        "standard_path": "standard_agent"
    }
)

# Both agents can use tools
workflow.add_edge("vip_agent", "tools")
workflow.add_edge("standard_agent", "tools")

# Tools return to respective agents
workflow.add_edge("tools", END)# Compile

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="support_agent_with_tools_anduser_tier.png")
# Run
# Test VIP
print("Testing VIP user:")
result = app.invoke({
    "messages": [HumanMessage(content="I'm a VIP customer, Check order ORD123 status")],
    "should_escalate": False,
    "issue_type": "",
    "user_tier": ""
})


print("\n" + "="*50)
for msg in result["messages"]:
    if hasattr(msg, 'content'):
        print(f"{msg.type}: {msg.content}")
# create a ticket for the issue
print("*************************************************")
print("Result: ", result)
print("*************************************************")

# Test Standard
print("\nTesting Standard user:")
result = app.invoke({
    "messages": [HumanMessage(content="Check my order ORD123 please")],
    "should_escalate": False,
    "issue_type": "",
    "user_tier": ""
})

print("\n" + "="*50)
for msg in result["messages"]:
    if hasattr(msg, 'content'):
        print(f"{msg.type}: {msg.content}")
print("*************************************************")
print("Result: ", result)
print("*************************************************")



