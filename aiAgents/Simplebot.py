from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

#This code is a simple bot without any Conversation memory (i.e. single msg - single response)
load_dotenv()

class schema(TypedDict):
    message:list[HumanMessage]

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def process(state:schema)->schema:
    response=llm.invoke(state["message"])
    print(f"AI: {response.content}")
    return state

graph=StateGraph(schema)
graph.add_node("process_node",process)
graph.add_edge(START,"process_node")
graph.add_edge("process_node",END)
app=graph.compile()

userinput= input("Enter Your Message: ")
while(userinput!='exit'):
    app.invoke({"message":[HumanMessage(content=userinput)]})
    userinput= input("Enter Your Message: ")


