# Code Explorer

## Purpose

The Code Explorer repository provides a web-based application that helps users explore and understand code repositories. It leverages large language models (LLMs) and embedding models to provide detailed insights and answers to coding questions. The main features of the application include:

- Interactive web interface using Streamlit
- Integration with Neo4j vector database for storing and retrieving code documents
- Customizable prompt templates and output parsers for the agent
- Support for various LLMs and embedding models including OpenAI, Ollama, and AWS

## Prerequisites

Before setting up the application, ensure you have the following prerequisites:

- Docker and Docker Compose installed
- Python 3.8 or higher
- Git

## Installation

To install and set up the application, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/arjuuuuunnnnn/CodeExplorer.git
   cd CodeExplorer
   ```

2. Set up the environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy the example environment file and configure it:
   ```bash
   cp .example.env .env
   ```

## Configuration

The application requires several environment variables to be configured. These variables are defined in the `.example.env` file. Below is a description of each variable:

- `LLM`: The large language model to use (e.g., `codellama:7b-instruct`, `gpt-4`, `gpt-3.5`, `claudev2`)
- `EMBEDDING_MODEL`: The embedding model to use (e.g., `ollama`, `sentence_transformer`, `openai`, `aws`)
- `NEO4J_URI`: The URI for the Neo4j database
- `NEO4J_USERNAME`: The username for the Neo4j database
- `NEO4J_PASSWORD`: The password for the Neo4j database
- `OLLAMA_BASE_URL`: The base URL for the Ollama service
- `OPENAI_API_KEY`: The API key for OpenAI (required if using OpenAI LLM or embedding model)
- `AWS_ACCESS_KEY_ID`: The AWS access key ID (required if using AWS Bedrock LLM or embedding model)
- `AWS_SECRET_ACCESS_KEY`: The AWS secret access key (required if using AWS Bedrock LLM or embedding model)
- `AWS_DEFAULT_REGION`: The AWS region (default is `us-east-1`)

## Usage

To run the application, use Docker Compose:

1. Start the services:
   ```bash
   docker-compose up
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the Code Explorer application.

3. Interact with the application by selecting a language, specifying a directory containing code files, and asking coding questions.

## Troubleshooting

If you encounter any issues, consider the following troubleshooting steps:

- Ensure all environment variables are correctly configured in the `.env` file.
- Check the Docker Compose logs for any error messages.
- Verify that the Neo4j database is running and accessible.
- Ensure the specified directory contains code files with the correct file extensions.

For further assistance, refer to the GitHub repository or open an issue.

