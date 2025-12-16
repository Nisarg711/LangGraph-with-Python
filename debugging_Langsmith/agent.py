from typing import Annotated,TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv 
load_dotenv()

os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="FIRSTPROJECT"

class schema(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


#GRaph with tool calls
def make_graph():
    @tool
    def add(a:int,b:int)->int:
        """Add two numbers"""
        return a+b

    tools=[add]
    tool_node=ToolNode(tools)
    llm=ChatGroq(model='llama-3.1-8b-instant')
    llm_bind_tools=llm.bind_tools(tools)


    def llm_func(state:schema)->schema:
        return {"messages":[llm_bind_tools.invoke(state["messages"])]}

    def where_to_go(state:schema)->str:
        if not state["messages"][-1].tool_calls:
            return "end"
        return "tools"


    graph=StateGraph(schema)
    graph.add_node("llm_node",llm_func)
    graph.set_entry_point("llm_node")
    graph.add_node("tool_node",tool_node)
    graph.add_conditional_edges("llm_node",where_to_go,
                                {
                                    "end":END,
                                    "tools":"tool_node"
                                })
    graph.add_edge("tool_node","llm_node")
    app=graph.compile()
    return app

tool_agent=make_graph()
