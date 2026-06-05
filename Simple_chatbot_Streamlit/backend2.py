from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, Annotated,List
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import START, END,StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv  
import os
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

class state(TypedDict):
    message:Annotated[List[BaseMessage],add_messages]
    title:str

llm=ChatGroq(model='llama-3.1-8b-instant')
def llm_call(state:state)->state:
    res=llm.invoke(state["message"])
    return {"message":[res],"title": state.get("title","New Chat")}

def generate_title(state: state):

    if state.get("title") and state.get("title") != "New Chat":
        return {}

    first_user_message = None

    for msg in state["message"]:
        if isinstance(msg, HumanMessage):
            first_user_message = msg.content
            break

    if not first_user_message:
        return {}

    prompt = f"""
    Generate a short chat title in max 4 words.

    User message:
    {first_user_message}
    """

    res = llm.invoke(prompt)

    return {
        "title": res.content.strip()
    }

graph = StateGraph(state)

graph.add_node("generate_title", generate_title)
graph.add_node("llm_node", llm_call)

graph.add_edge(START, "generate_title")
graph.add_edge("generate_title", "llm_node")
graph.add_edge("llm_node", END)

config={'configurable':{"thread_id":"1"}}

memory=InMemorySaver()
app=graph.compile(checkpointer=memory)

# # gen=app.stream({'message':[HumanMessage(content="What is the capital of France?")]},config=config,stream_mode="messages")
# for msg_chunk,metadata in app.stream({'message':[HumanMessage(content="What is the capital of France?")]},config=config,stream_mode="messages"):
#     print(msg_chunk.content)