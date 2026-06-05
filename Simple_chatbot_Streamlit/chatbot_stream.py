import streamlit as st
from langchain_core.messages import HumanMessage
from backend import app
st.title("Chatbot Interface")
st.subheader("Welcome to the Chatbot Interface! Feel free to ask anything.")

if 'message' not in st.session_state:
    st.session_state['message']=[]

for msg in st.session_state['message']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

config={"configurable":{"thread_id":"4"}}
userinp=st.chat_input("Type your message here...")
if userinp:
    # res=app.invoke({'message':[HumanMessage(content=userinp)]},config=config)
    st.session_state['message'].append({'role': 'user', 'content': userinp})
    with st.chat_message("user"):
        st.text(userinp)
  
    with st.chat_message("assistant"):
       ai_msg = st.write_stream(
            msg_chunk.content for msg_chunk,metadata in
            app.stream(
            {'message':[HumanMessage(content=userinp)]},
            config=config,
            stream_mode="messages"
        ))
    st.session_state['message'].append({'role':'assistant','content':ai_msg})

