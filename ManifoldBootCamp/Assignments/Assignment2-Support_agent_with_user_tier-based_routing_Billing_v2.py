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


from langchain_core.messages import AIMessage, ToolMessage

'''def should_continue(state: SupportState) -> str:
    last_message = state["messages"][-1]

    # If no tool calls, we are done
    if not getattr(last_message, "tool_calls", None):
        return "end"

    # Route back to correct agent
    if state.get("issue_type") == "billing":
        return "billing_agent"
    elif state.get("user_tier") == "vip":
        return "vip_agent"
    else:
        return "standard_agent"

'''

def should_continue(state: SupportState) -> str:
    last_message = state["messages"][-1]

    # If the last message is a ToolMessage,
    # we MUST go back to an agent to consume it
    if isinstance(last_message, ToolMessage):
        if state.get("issue_type") == "billing":
            return "billing_agent"
        elif state.get("user_tier") == "vip":
            return "vip_agent"
        else:
            return "standard_agent"

    # If the last message is an AIMessage WITH tool calls → tools already ran
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if state.get("issue_type") == "billing":
            return "billing_agent"
        elif state.get("user_tier") == "vip":
            return "vip_agent"
        else:
            return "standard_agent"

    # Otherwise, AIMessage without tool calls = final answer
    return "end"


## Extent the Support Agent

def classify_issue_node(state: SupportState):
    """Check message and set issue type to billing or general"""

    messages = state["messages"]
    first_message = messages[0].content.lower()

    if "refund" in first_message or "billing" in first_message:
        return {"issue_type": "billing"}
    else:
        return {"issue_type": "general"}

def route_after_classify(state: SupportState) -> str:
    if state.get("issue_type") == "billing":
        return "billing_path"
    elif state.get("user_tier") == "vip":
        return "vip_path"
    else:
        return "standard_path"


def route_back_to_agent(state: SupportState) -> str:
    # Route back to the agent that should consume the tool result
    if state.get("issue_type") == "billing":
        return "billing_agent"
    elif state.get("user_tier") == "vip":
        return "vip_agent"
    else:
        return "standard_agent"

# Build graph
workflow = StateGraph(SupportState)
workflow.add_node("check_tier", check_user_tier_node)
workflow.add_node("classify_issue", classify_issue_node)      # NEW
workflow.add_node("vip_agent", vip_agent_node)
workflow.add_node("standard_agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("billing_agent", billing_agent_node)

# Set entry point
workflow.set_entry_point("check_tier")

# Example: first decide issue type, then route
workflow.add_edge("check_tier", "classify_issue")

workflow.add_conditional_edges(
    "classify_issue",
    route_after_classify,
    {
        "billing_path": "billing_agent",
        "vip_path": "vip_agent",
        "standard_path": "standard_agent",
    }
)

# Both agents can use tools
workflow.add_edge("vip_agent", "tools")
workflow.add_edge("standard_agent", "tools")
workflow.add_edge("billing_agent", "tools")

# Tools return to respective agents
#workflow.add_edge("tools", END)# Compile


workflow.add_conditional_edges(
    "tools",
    should_continue,
    {
        "billing_agent": "billing_agent",
        "vip_agent": "vip_agent",
        "standard_agent": "standard_agent",
        "end": END,
    }
)

app = workflow.compile()
#app.get_graph().draw_mermaid_png(output_file_path="support_agent_with_tools_anduser_tier.png")
app.get_graph().draw_mermaid_png(output_file_path="support_agent_with_tools_and_user_tier_billing_agent.png")
# Run
# Test VIP
print("Testing VIP user:")
result = app.invoke({
    "messages": [HumanMessage(content="I'm a VIP customer, Check order ORD123 status and issue a refund")],
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
    "messages": [HumanMessage(content="Check my order ORD123 and issue my refund")],
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



