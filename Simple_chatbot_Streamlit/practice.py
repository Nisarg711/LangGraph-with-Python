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



userinp=st.chat_input("Type your message here...")
if userinp:
    st.session_state['message'].append({'role': 'user', 'content': userinp})
    # with st.chat_message("user"):
    #     st.text(userinp)
    st.session_state['message'].append({'role': 'assistant', 'content': "This is a response from the AI: " + userinp})
    with st.chat_message("ai"):
        st.text("This is a response from the AI: " + userinp)

#When user presses enter, execution goes to top, and then when it reaches userinp, 
#it is initialised with the new input, and then it is added to the session state, and
#  then it is displayed in the chat interface. so for loop is placed here.


for msg in st.session_state['message']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])