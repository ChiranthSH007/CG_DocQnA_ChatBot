import streamlit as st
import os
import model

# Set up the page layout
st.set_page_config(page_title="Chat and File Upload", layout="wide")
st.title("Doc QnA ChatBot")
# Create two columns with a 30-70 percent width split
col1,spacer, col2 = st.columns([0.3,0.05, 0.7])

bot=model.RAGPDFBot()

# Function to display chat bubbles
def display_chat_bubbles(messages):
    for message in messages:
        if message['role'] == 'user':
            st.chat_message(message['role']).markdown(f"<div style='background-color:#DCF8C6;padding:10px;border-radius:10px;color:black'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.chat_message(message['role']).markdown(f"<div style='background-color:#f1f1f1;padding:10px;border-radius:10px;color:black'>{message['content']}</div>", unsafe_allow_html=True)


def initialize_model(filePath):
    # query = query_text.value
    top_k = 2
    chunk_size = 500
    overlap = 50
    max_length = 128
    # rag_off = rag_off_checkbox.value
    temp = 0.7
    model_loaded = False
    if model_loaded==False:
        st.write("Loading Model...")
        bot.load_model(max_length=max_length, repeat_penalty=1.50, top_k=top_k, temp=temp)
        model_loaded=True
        #build the vector database
        st.write("Building vector DB...")
        bot.build_vectordb(chunk_size = chunk_size, overlap = overlap,file_path=filePath)
        st.write("Model and Vector DB Build Done!!")



def retrive(input):
    bot.retrieval(user_input = input )
    return bot.inference()


# First Column - File Upload
with col1:
    st.header("Upload File")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        save_dir = "DocumentQnAChatBot/"  # You can specify any directory
        
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the uploaded file to the specified directory
        file_path = os.path.join(save_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Save the file as bytes
        
        # Optionally, you can also show the absolute path of the saved file
        abs_file_path = os.path.abspath(file_path)
        # st.write(f"Absolute path: {abs_file_path}")
        initialize_model(abs_file_path)
        
   

# Second Column - Chat Window
with col2:
    st.header("Chat Window")
    
    # Define some initial messages for the chat window
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
    
    # Display existing chat messages
    display_chat_bubbles(st.session_state['messages'])
    
    if 'text_value' not in st.session_state:
        st.session_state.text_value = ''

    # User input for new message
    user_input = st.text_input("Your message:", value=st.session_state.text_value)
    send_button = st.button("Send")
    
    # When the user presses enter or the button is clicked, add the message
    if send_button or user_input:
        st.session_state['messages'].append({"role": "assistant", "content": "I'm processing your input..."})
        st.session_state['messages']=[{"role": "assistant", "content": retrive(user_input)}]
        display_chat_bubbles(st.session_state['messages'])
        st.session_state['messages'].append({"role": "user", "content": user_input})
        st.session_state.text_value = '' # reset the input field
