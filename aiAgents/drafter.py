from typing import TypedDict, Annotated,Sequence 
from langgraph.graph import StateGraph,END
from langchain_core.messages import BaseMessage,SystemMessage,ToolMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

load_dotenv()

global_content = "" #Its a global variable 

@tool
def update(content:str)->str:
    """This is a tool to update global content"""
    global global_content
    global_content = content
    return f"Successfully updated content to {global_content}"

@tool
def save(filename:str)->str:
    """To save the content"""
    if(not filename.endswith(".txt")):
        filename=filename+".txt"
    global global_content
    with open(filename,"w") as f:
        f.write(global_content)
    return f"Successfully saved content to {filename}"

tools=[update,save]
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7).bind_tools(tools=tools)

class schema(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

def our_agent(state: schema) -> schema:
    sys_prompt = SystemMessage(
        "You are an expert drafter. "
        "You help users draft documents based on their inputs. "
        "You can use the tools to update the content or save it. "
        "Following is the file Content so far: " + global_content
    )

    if not state["messages"]:
        userinput = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=userinput)

    else:
        userinput = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {userinput}")
        user_message = HumanMessage(content=userinput)

    messages = [sys_prompt] + list(state["messages"]) + [user_message]

    ai_msg = llm.invoke(messages)
    print(f"\n🤖 AI: {ai_msg.content}")
    return {"messages": [ai_msg]}


  
def should_continue(state: schema) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            if msg.content.startswith("Successfully saved"):
                return "end"
            return "continue"

    return "continue"



graph=StateGraph(schema)
graph.add_node("agent_node",our_agent)
graph.set_entry_point("agent_node")
tool_node=ToolNode(tools=tools)
graph.add_node("tool_node",tool_node)  
graph.add_edge("agent_node","tool_node")#In previous code, we didn't need this edge since
                #the conditional check was at agent node itself while in this case, its at toolnode

graph.add_conditional_edges("tool_node",should_continue,
                            {
                                "continue":"agent_node",
                                "end":END
                            })


app=graph.compile()

result=app.invoke({"messages":[]})
print(result)







    
