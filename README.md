# Document Chat API

The Document Chat API is a powerful tool that allows users to upload documents and chat with them, integrating directly with a SQL database to fetch relevant data for answering questions. This API uses advanced AI techniques to parse and understand both the document contents and database records to provide comprehensive answers.

## Features

- **Document Interaction**: Users can upload documents and ask questions based on the content.
- **SQL Database Integration**: Connects to a SQL database to use stored data in answering user queries.
- **Flexible Deployment**: Includes a Dockerfile for containerized deployment.

## Prerequisites

- Docker installed on your machine (optional)
- Python 3.8 or higher if running locally

## Installation

### Running with Docker

1. **Clone the repository:**

```bash
git clone https://github.com/abdulzain6/Document-Chat-API.git
cd Document-Chat-API
```

2. **Build and run the Docker container:**

```bash
docker build -t document-chat-api .
docker run -p 8000:8000 document-chat-api
```

### Local Setup

1. **Install the required Python packages:**

```bash
pip install -r requirements.txt
```

## Configuration

Configure the API by setting the following environment variables:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
DB_HOST=your_database_host
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_NAME=your_database_name
DB_TABLE_NAMES=comma_separated_list_of_table_names
```

## Usage

Start the API server (if not using Docker):

```bash
python api.py
```

Users can now upload documents and interact via the API endpoints to ask questions and receive answers based on the uploaded documents and connected SQL database data.

## Contributing

Contributions are highly encouraged. Please fork this repository, make your changes, and submit a pull request.