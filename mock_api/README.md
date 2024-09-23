# Comprehensive RAG Benchmark (CRAG) Mock API

## Prerequisites

Before diving into the setup and usage of the CRAG Mock API, ensure you have the following prerequisites installed and set up on your system:
- Git (for cloning the repository)
- Python 3.10

## Installation Guide

### Setting Up Your Environment

First, clone the repository to your local machine using Git. Then, navigate to the repository directory and install the necessary dependencies:

```
cd mock_api
pip install -r requirements.txt
```

## Running the API Server

To launch the API server on your local machine, use the following Uvicorn command. This starts a fast, asynchronous server to handle API requests.

```
uvicorn server:app --reload
```

Access the API documentation and test the endpoints at `http://127.0.0.1:8000/docs`.

For custom server configurations, specify the host and port as follows:

```
uvicorn server:app --reload --host [HOST] --port [PORT]
```

## System Requirements

- **Supported OS**: Linux, Windows, macOS
- **Python Version**: 3.10
- See `requirements.txt` for a complete list of Python package dependencies.

## Python API Wrapper

For Python developers, the [/mock_api/apiwrapper/pycragapi.py](/mock_api/apiwrapper/pycragapi.py) provides a convenient way to interact with the API. An example usage is demonstrated in [/mock_api/apiwrapper/example_call.ipynb](/mock_api/apiwrapper/example_call.ipynb), showcasing how to efficiently integrate the API into your development workflow.
