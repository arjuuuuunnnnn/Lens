## Developing...

- Neo4j vector database for storing and retrieving code documents
- Customizable prompt templates and output parsers for the agent
- Support for various LLMs and embedding models

## Conf

The application requires several environment variables to be configured. These variables are defined in the `.example.env` file. Below is a description of each variable:

- `LLM`: The large language model to use (`codellama:7b-instruct`)
- `EMBEDDING_MODEL`: The embedding model to use (e.g., `ollama`, `sentence_transformer`)
- `NEO4J_URI`: The URI for the Neo4j database
- `NEO4J_USERNAME`: The username for the Neo4j database
- `NEO4J_PASSWORD`: The password for the Neo4j database
- `OLLAMA_BASE_URL`: The base URL for the Ollama service
