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

llm=ChatGroq(model='llama-3.1-8b-instant')
def llm_call(state:state)->state:
    res=llm.invoke(state["message"])
    return {"message":[res]}


graph=StateGraph(state)
graph.add_node("llm_node",llm_call)
graph.add_edge(START,"llm_node")
graph.add_edge("llm_node",END)

config={'configurable':{"thread_id":"1"}}

memory=InMemorySaver()
app=graph.compile(checkpointer=memory)