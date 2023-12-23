import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import DeepInfra
from langchain.chains import ConversationalRetrievalChain

# Set environment variables
load_dotenv()
deepinfra_api_token = st.secrets["DEEPINFRA_API_TOKEN"]
if deepinfra_api_token:
    os.environ["DEEPINFRA_API_TOKEN"] = deepinfra_api_token

# Set global constant
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Function to load the LLM based on the selected option
def load_llm(selected_llm):
    models = {
        "meta-llama/Llama-2-7b-chat-hf": {"temperature": 0.5, "repetition_penalty": 1.2, "max_new_tokens": 256, "top_p": 0.95},
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"temperature": 0.3, "repetition_penalty": 1.2, "max_new_tokens": 256, "top_p": 0.95},
        "mistralai/Mistral-7B-Instruct-v0.1": {"temperature": 0.1, "repetition_penalty": 1.2, "max_new_tokens": 256, "top_p": 0.95},
    }

    if selected_llm not in models:
        raise ValueError("Invalid LLM selected")

    return DeepInfra(model_id=f"{selected_llm}", model_kwargs=models[selected_llm])

st.set_page_config(layout="wide", page_title="Cognitut")

st.markdown("""
<style>
    /* side bar background*/ 
    [data-testid=stSidebar] {
        background-color: #E8C6FF;
        
        
        /* uploded file box*/ 
        div.uploadedFile.st-emotion-cache-12xsiil.e1b2p2ww5 {
            background-color: #FFC6FA;
            opacity:1;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("COGNITUT")
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: #FF2F2F; 
            background-color: #DAF7A6; 
            padding: 5px;
            z-index: 9999;  
        }
    </style>
    <div class="footer">Github Repository :  <a href='https://github.com/Rahul-INX/Cognitut'>Cognitut</a></div>
    """,
    unsafe_allow_html=True
)

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

# Display a loading message or placeholder while waiting for file upload
if not uploaded_file:
    st.warning("Please upload a CSV file first.")
else:
    # Model selection
    selected_llm = st.sidebar.selectbox("Select LLM", ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"], index=0)

    # Load LLM and chain only after file upload, tracking model changes
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    # Load the LLM initially or when the model selection changes
    global llm, chain
    if "llm" not in st.session_state or st.session_state.get("selected_llm") != selected_llm:
        llm = load_llm(selected_llm)
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
        st.session_state["selected_llm"] = selected_llm

    # Chat functionality
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = [f"Hello! Ask me anything about: '{uploaded_file.name}'"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # Container for the chat history
    response_container = st.container()
    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="INPUT YOUR QUERY", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',avatar_style="big-smile",seed="felix")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts",seed="aneka")
