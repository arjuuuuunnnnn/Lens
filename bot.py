import os
import streamlit as st
from streamlit.logger import get_logger
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from chains import (
    configure_llm_only_chain,
    get_qa_rag_chain
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.text_splitter import Language
from agent import get_agent_executor
from db import process_documents

# set page title
st.set_page_config(
    page_title="Lens",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "GitHub: https://github.com/arjuuuuunnnnn/Lens"
    }
)

load_dotenv(".env")

# Configuration
url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "password")
gemini_api_key = os.getenv("GOOGLE_API_KEY")
embedding_model_name = os.getenv("EMBEDDING_MODEL", "google")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

logger = get_logger(__name__)

@st.cache_resource
def init_llm():
    # Create Gemini LLM
    llm = GoogleGenerativeAI(
        model=gemini_model,
        google_api_key=gemini_api_key,
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=2048,
    )
    return llm

@st.cache_resource
def init_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )
    dimension = 768  # Google's embedding dimension
    return embeddings, dimension

llm = init_llm()
embeddings, dimension = init_embeddings()

@st.cache_resource
def get_llm_chain():
    chain = configure_llm_only_chain(llm)
    return chain

@st.cache_resource
def process_directory(language, directory, count) -> (str, Neo4jVector):
    error, vectorstore = process_documents(language, directory, embeddings=embeddings, 
                                          url=url, username=username, password=password)
    return (error, vectorstore)

@st.cache_resource
def get_qa_chain(_vectorstore, count):
    qa = get_qa_rag_chain(_vectorstore, llm)
    return qa

@st.cache_resource
def get_agent(_qa, count):
    qa = get_agent_executor(_qa, llm)
    return qa

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def main():
    qa = None
    agent = None
    llm_chain = get_llm_chain()

    if "language" not in st.session_state:
        st.session_state["language"] = None
    if "directory" not in st.session_state:
        st.session_state["directory"] = None
    if "detailedMode" not in st.session_state:
        st.session_state["detailedMode"] = True
    if "vectorstoreCount" not in st.session_state:  # only incremented to reset cache for processDocuments()
        st.session_state["vectorstoreCount"] = 0
    if "qaCount" not in st.session_state:           # only incremented to reset cache for get_qa_rag_chain()
        st.session_state["qaCount"] = 0    
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    # sidebar
    with st.sidebar:
        # Convert enum values to a list of strings
        languages_list = [lang.value for lang in Language]
        default_index = languages_list.index(Language.PYTHON)
        languageSelected = st.selectbox(
            'Select language',
            languages_list,
            index=default_index
        )

        currentPath = os.getcwd()
        directory = st.text_input('Enter folder path', currentPath)
        directory = directory.strip()

        # Add a section for Gemini model selection
        gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        selected_model = st.selectbox(
            'Select Gemini model',
            gemini_models,
            index=0
        )
        
        # Update the model if changed
        if selected_model != gemini_model:
            os.environ["GEMINI_MODEL"] = selected_model
            st.session_state["qaCount"] += 1  # Force reinitialization
            st.experimental_rerun()

        processBtnClicked = st.button('Process files')
        if processBtnClicked:
            if not os.path.exists(directory):
                st.error("Path doesn't exist!")
            else:
                if isinstance(directory, str) and directory:
                    st.session_state["language"] = languageSelected
                    st.session_state["directory"] = directory
                    st.session_state["vectorstoreCount"] += 1
                    st.session_state["qaCount"] += 1
                    st.session_state["user_input"] = []
                    st.session_state["generated"] = []

        # show folder selected
        if st.session_state["directory"]:
            st.code(st.session_state["directory"])

            error, vectorstore = process_directory(st.session_state["language"], st.session_state["directory"], st.session_state["vectorstoreCount"])

            if error:
                st.error(error)
            elif vectorstore:
                qa = get_qa_chain(vectorstore, st.session_state["qaCount"])
                agent = get_agent(qa, st.session_state["qaCount"])

                # show clear chat history button
                clearMemoryClicked = st.button("🧹 Reset chat history")
                if clearMemoryClicked:
                    st.session_state["qaCount"] += 1
                    st.session_state["user_input"] = []
                    st.session_state["generated"] = []

                    qa = get_qa_rag_chain(vectorstore, st.session_state["qaCount"])
                    agent = get_agent(qa, st.session_state["qaCount"])

                # show toggle to switch between qa and agent mode
                detailedMode = st.toggle('Detailed mode', value=True)
                st.session_state["detailedMode"] = detailedMode

    # load previous chat history
    if st.session_state["generated"]:
        size = len(st.session_state["generated"])
        # Display all exchanges
        for i in range(0, size):
            with st.chat_message("user"):
                st.write(st.session_state["user_input"][i])
            with st.chat_message("assistant"):
                st.write(st.session_state["generated"][i])

    # user chat
    user_input = st.chat_input("What coding issue can I help you resolve today?")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
            st.session_state["user_input"].append(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                stream_handler = StreamHandler(st.empty())
                if qa:
                    if st.session_state["detailedMode"]:
                        print("Using QA")
                        result = qa(
                            {"question": user_input},
                            callbacks=[stream_handler]
                        )
                        answer = result["answer"]
                    else:
                        print("Using Agent")
                        result = agent(
                            {"input": user_input},
                            callbacks=[stream_handler]
                        )
                        answer = result["output"]
                else:
                    print("Using LLM only")
                    answer = llm_chain(
                        {"question": user_input},
                        callbacks=[stream_handler]
                    )

                st.session_state["generated"].append(answer)


if __name__ == "__main__":
    main()
