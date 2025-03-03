import os
from streamlit.logger import get_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector

logger = get_logger(__name__)

def clear_neo4j_data(url, username, password, node_label):
    """Custom function to clear Neo4j data properly"""
    try:
        driver = GraphDatabase.driver(url, auth=(username, password))
        with driver.session() as session:
            # Use the correct Cypher syntax for deletion
            session.run(f"MATCH (n:`{node_label}`) DETACH DELETE n")
        driver.close()
        return True
    except Exception as e:
        logger.error(f"Error clearing Neo4j data: {e}")
        return False

def create_vector_index(url, username, password, index_name, node_label, dimension, similarity_metric="cosine"):
    """Custom function to create vector index with correct syntax"""
    try:
        driver = GraphDatabase.driver(url, auth=(username, password))
        with driver.session() as session:
            # Check if Neo4j supports vector indexes (Neo4j 5.11+ or AuraDB)
            neo4j_version = session.run("CALL dbms.components() YIELD name, versions RETURN versions[0] as version").single()["version"]
            major_version = int(neo4j_version.split('.')[0])
            
            # For Neo4j 5.11+ use proper vector index syntax
            if major_version >= 5:
                # First drop the index if it exists
                try:
                    session.run(f"DROP INDEX {index_name} IF EXISTS")
                except:
                    pass
                
                # Create the vector index with proper syntax
                query = f"""
                CREATE INDEX {index_name} FOR (n:`{node_label}`)
                ON (n.embedding)
                OPTIONS {{
                  indexProvider: 'vector',
                  indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: '{similarity_metric}'
                  }}
                }}
                """
                session.run(query)
            else:
                logger.error("Neo4j version does not support vector indexes. Please use Neo4j 5.11+ or AuraDB")
                return False
        driver.close()
        return True
    except Exception as e:
        logger.error(f"Error creating vector index: {e}")
        return False

def process_documents(language, directory, embeddings=None, url=None, username=None, password=None) -> (str, Neo4jVector):
    print("File chunking begins...", language, directory)
    
    # Create a dictionary mapping languages to file extensions
    language_suffix_mapping = {
        Language.CPP: ".cpp",
        Language.GO: ".go",
        Language.JAVA: ".java",
        Language.KOTLIN: ".kt",
        Language.JS: ".js",
        Language.TS: ".ts",
        Language.PHP: ".php",
        Language.PROTO: ".proto",
        Language.PYTHON: ".py",
        Language.RST: ".rst",
        Language.RUBY: ".rb",
        Language.RUST: ".rs",
        Language.SCALA: ".scala",
        Language.SWIFT: ".swift",
        Language.MARKDOWN: ".md",
        Language.LATEX: ".tex",
        Language.HTML: ".html",
        Language.SOL: ".sol",
        Language.CSHARP: ".cs",
    }
    # Get the corresponding suffix based on the selected language
    suffix = language_suffix_mapping.get(language, "")
    print("language file extension:", suffix)
    loader = GenericLoader.from_filesystem(
        path=directory,
        glob="**/*",
        suffixes=[suffix],
        parser=LanguageParser(language=language, parser_threshold=500)
    )
    documents = loader.load()
    print("Total documents:", len(documents))
    if len(documents) == 0:
        return ("0 documents found", None)
    text_splitter = RecursiveCharacterTextSplitter.from_language(language=language, 
                                                               chunk_size=5000, 
                                                               chunk_overlap=500)
    chunks = text_splitter.split_documents(documents)
    print("Chunks:", len(chunks))
    hashStr = "myHash" # str(abs(hash(directory)))
    
    # First clear any existing data
    node_label = f"node_{hashStr}"
    index_name = f"index_{hashStr}"
    
    clear_success = clear_neo4j_data(url, username, password, node_label)
    if not clear_success:
        return ("Error clearing existing Neo4j data", None)
    
    # Store the chunks part in db (vector)
    try:
        # Manually create the vector index with correct syntax
        dimension = 768  # Google's embedding dimension
        index_created = create_vector_index(url, username, password, index_name, node_label, dimension)
        if not index_created:
            return ("Error creating vector index", None)
        
        # Modified approach to use Neo4jVector
        vectorstore = Neo4jVector(
            embedding=embeddings,
            url=url,
            username=username,
            password=password,
            index_name=index_name,
            node_label=node_label,
            text_node_property="text",
            embedding_node_property="embedding",
        )
        
        # Add documents to the vector store
        vectorstore.add_documents(chunks)
        
        print("Files are now chunked up")
        return (None, vectorstore)
    except Exception as e:
        error_msg = f"Error creating vector store: {str(e)}"
        logger.error(error_msg)
        return (error_msg, None)
