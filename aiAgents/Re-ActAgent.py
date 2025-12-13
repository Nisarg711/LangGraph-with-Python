from typing import TypedDict, Annotated,Sequence 
from langgraph.graph import StateGraph,END
from langchain_core.messages import BaseMessage,SystemMessage,ToolMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

load_dotenv()

@tool
def add(a:int,b:int)->int:
    """This is a tool to add two numbers"""
    return a+b

tools=[add]
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7).bind_tools(tools=tools)


class schema(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


def model_call(state:schema)->schema:
    sys_prompt=SystemMessage("You are my helpful assistant!")
    response=llm.invoke([sys_prompt] + state["messages"])
    return {"messages":[response]}


def should_continue(state:schema)->str:
    lastmsg=state["messages"][-1]
    if not lastmsg.tool_calls:
        return "end"
    return "continue"

graph=StateGraph(schema)
graph.add_node("agent",model_call)
graph.set_entry_point("agent")
tool_node=ToolNode(tools=tools)
graph.add_node("tools",tool_node)
graph.add_conditional_edges("agent",should_continue,
                            {
                                "continue":"tools",
                                "end":END
                            })
graph.add_edge("tools","agent")

app=graph.compile()


userinp=input("Enter user message: ")

while userinp!="exit":
    result = app.invoke({"messages": [HumanMessage(content=userinp)]})   
    print(result)
    # Print AI messages
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print("AI Message:", msg.content)
    
    # Print tool messages
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print("Tool Message:", msg.content)
    
    userinp=input("Enter user message: ")
    
