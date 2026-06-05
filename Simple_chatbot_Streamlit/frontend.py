import streamlit as st
from langchain_core.messages import HumanMessage
from backend import app
st.title("Chatbot Interface")
st.subheader("Welcome to the Chatbot Interface! Feel free to ask anything.")
# with st.chat_message("user"):
#     st.text("Hello, This is my first time using streamlit.")


# with st.chat_message("ai"):
#     st.text("Hello! Welcome to Streamlit. How can I assist you today?")

# userinput=st.chat_input("Type your message here...")
# if userinput:
#     with st.chat_message("user"):
#         st.text(userinput)

#st.session_state is a dictonary that doesn't change when script rexecutes, 
# so we can store the messages in it and it will persist across reruns of the script.
#But it will reset when we refresh browser page.
if 'message' not in st.session_state:
    st.session_state['message']=[]



# userinp=st.chat_input("Type your message here...")
# if userinp:
#     st.session_state['message'].append({'role': 'user', 'content': userinp})
#     # with st.chat_message("user"):
#     #     st.text(userinp)
#     st.session_state['message'].append({'role': 'assistant', 'content': "This is a response from the AI: " + userinp})
    # with st.chat_message("ai"):
    #     st.text("This is a response from the AI: " + userinp)

for msg in st.session_state['message']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

config={"configurable":{"thread_id":"4"}}
userinp=st.chat_input("Type your message here...")
if userinp:
    res=app.invoke({'message':[HumanMessage(content=userinp)]},config=config)
    st.session_state['message'].append({'role': 'user', 'content': userinp})
    with st.chat_message("user"):
        st.text(userinp)
    st.session_state['message'].append({'role':'assistant','content':res['message'][-1].content})
    with st.chat_message("assistant"):
        st.text(res['message'][-1].content)