from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, Annotated,List
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import START, END,StateGraph
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv  
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sqliteDB")

connectionobj = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)

memory=SqliteSaver(conn=connectionobj)
# We need to create a sqlite db and connect this checkptr to it
#Since we are going to handle multiple threads, and sqlite db uses only one thread and can use
#this db in same thread, so when set as False, we will be able to use it across multiple thread
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

# memory=InMemorySaver()

app=graph.compile(checkpointer=memory)



# res=app.invoke({'message':[HumanMessage(content='Hi there myself NB')]},config=config)
# res=app.invoke({'message':[HumanMessage(content='Hi what was my last question')]},config=config)
# print(res['message'][-1])
# res=app.invoke({'message':[HumanMessage(content="list out our conversations so far")]},config=config)
# print(res['message'][-1].content)

# genobj=memory.list(None) #Tells the number of checkpoints in a particular thread or all threads depending on
            #what we specify, when None para pass, it means give all checkpointers

def retrieve_all_threads():
    allthreads=set()
    # print(genobj)  #Gives <generator object SqliteSaver.list at 0x124f17c40>
    for checkptr in memory.list(None):
        # print(checkptr.config['configurable']['thread_id'])
        allthreads.add(checkptr.config['configurable']['thread_id'])
    lis=list(allthreads)
    return lis
