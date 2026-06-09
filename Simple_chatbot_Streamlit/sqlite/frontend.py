import streamlit as st
from langchain_core.messages import HumanMessage
from backend_db import app,retrieve_all_threads
import uuid

#Utitlity function to generate unique thread ids for each chat session
def generate_thread_id():
    return str(uuid.uuid4())

def clear_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    res=app.get_state(config={"configurable":{"thread_id": thread_id}})
    return res.values.get('message', [])
    

#**********************Session Manager********************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()

# if 'chat_threads' not in st.session_state:
#     st.session_state['chat_threads']={}
#earlier we didn;t have permanent storage so we initialzed empty {} everytime
#Now here we needa query the db and get to know ki kitni threads hai,and save each in this

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=retrieve_all_threads() 



add_thread(st.session_state['thread_id'])

#********************************************************************************


#*********************************SideBar UI**************************************


st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("Start New Chat"):
    clear_chat()

st.sidebar.subheader("Chat History")
for thid in reversed(st.session_state['chat_threads']):
    if st.sidebar.button(str(thid),key=thid):
        st.session_state['thread_id']=thid
        
        msgs=load_conversation(thid)        #here the msg we get are in form of list of 
                                         #humanMsg and AIMessage from which use.content to get
        temp_msg=[]
        for msg in msgs:
            if isinstance(msg,HumanMessage):
                role="user"
            else:
                role="assistant"
            temp_msg.append({'role': role, 'content': msg.content})
        st.session_state['message_history']=temp_msg




#***********************************************************************************

for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

config={"configurable":{"thread_id":st.session_state['thread_id']}}
userinp=st.chat_input("Type your message here...")
if userinp:
    # res=app.invoke({'message':[HumanMessage(content=userinp)]},config=config)
    st.session_state['message_history'].append({'role': 'user', 'content': userinp})
    with st.chat_message("user"):
        st.text(userinp)
  
    with st.chat_message("assistant"):
       ai_msg = st.write_stream(
        msg_chunk.content for msg_chunk, metadata in
        app.stream(
        {'message': [HumanMessage(content=userinp)]},
        config=config,
        stream_mode="messages"
        )
        if metadata.get("langgraph_node") == "llm_node"
        and hasattr(msg_chunk, "content")
        and msg_chunk.content
)
       state = app.get_state(config=config)
       
    st.session_state['message_history'].append({'role':'assistant','content':ai_msg})
    st.rerun()

