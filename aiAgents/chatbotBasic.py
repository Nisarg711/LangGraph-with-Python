from typing import TypedDict,Union
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import StateGraph, START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

#This code is a simple bot without any Conversation memory (i.e. single msg - single response)
load_dotenv()

class schema(TypedDict):
    message:list[Union[HumanMessage,AIMessage]]

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def process(state:schema)->schema:
    response=llm.invoke(state['message'])
    state['message'].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    return state

graph=StateGraph(schema)
graph.add_node("process_node",process)
graph.add_edge(START,"process_node")
graph.add_edge("process_node",END)

app=graph.compile()

conversation_history=[]
userinput=input("Enter Your Message: ")

while(userinput!='exit'):
    conversation_history.append(HumanMessage(content=userinput))
    result=app.invoke({"message":conversation_history})
    # print(result["message"])
    conversation_history=result['message']
    userinput=input("Enter Your Message: ")

    #Output found as follows

# Enter Your Message: Hi there my name is Nisarg
# AI: Hi Nisarg! It's nice to meet you.

# I'm an AI, and I'm here to help you. How can I assist you today?
# Enter Your Message: what is my name
# AI: Your name is Nisarg.
# Enter Your Message: Are you sure?
# AI: Yes, I am sure. You told me your name in your very first message: "Hi there my name is Nisarg".
# Enter Your Message: what are you?
# AI: I am a large language model, trained by Google.