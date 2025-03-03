from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from typing import List, Any
from utils import BaseLogger
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.get("google_api_key", "")
        )
        dimension = 768
        logger.info("Embedding: Using Google Embedding Model")
    else:
        # Default to Google embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.get("google_api_key", "")
        )
        dimension = 768
        logger.info("Embedding: Using Google Embedding Model (default)")
    
    return embeddings, dimension


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering programming questions.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    """
    human_template = "{question}"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),        # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"),          # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(human_template)    # Where the human input will injected
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        verbose=False,
        memory=memory,
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any]
    ) -> str:    
        answer = chain.invoke(user_input, config={"callbacks": callbacks})["text"]
        return answer

    return generate_llm_output


def get_qa_rag_chain(_vectorstore, llm):
    # Create qa RAG chain
    system_template = """ 
    Use the following pieces of context to answer the question at the end.
    The context contains code source files which can be used to answer the question as well as be used as references.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    Generate concise answers with references to code source files at the end of every answer.
    """
    user_template = "Question:```{question}```"
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template), # The persistent system prompt
        HumanMessagePromptTemplate.from_template(user_template),    # Where the human input will injected
    ])
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=chat_prompt,
    )
    qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
        return_source_documents=True
    )

    return qa
