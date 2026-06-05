from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,BaseMessage
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from typing import Annotated, List,TypedDict
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dotenv import load_dotenv
import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"]="FIRSTPROJECT"


class state(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def make_graph():
    @tool
    def add(a:int,b:int)->int:
        """Add two numbers together and return the sum."""
        return a+b

    tools=[add]
    llm=ChatGroq(model="llama-3.1-8b-instant").bind_tools(tools)

    def llm_model(state:state)->state:
        res=llm.invoke(state["messages"])
        return {"messages":[res]}

    def where_to_go(state:state)->str:
        last_msg=state["messages"][-1]
        if last_msg.tool_calls:
            return "call_tool"
        return "end"
    
    graph=StateGraph(state)
    graph.add_node("llm",llm_model)
    graph.add_edge(START,"llm")
    tool_node=ToolNode(tools)
    graph.add_node("tool_node",tool_node)
    graph.add_conditional_edges("llm",where_to_go,
                                {
                                 "call_tool":"tool_node",
                                 "end":END   
                                })
    graph.add_edge("tool_node","llm")
    app=graph.compile()
    return app

tool_agent=make_graph()

