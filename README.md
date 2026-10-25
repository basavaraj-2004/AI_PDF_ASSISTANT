# AI PDF Assistant

An AI-powered PDF question-answering application that lets users upload PDF documents and ask questions in natural language.

## Overview

AI PDF Assistant combines a React frontend with a FastAPI backend and Ollama to help users retrieve information from their PDF documents.

## Features

- Upload PDF documents.
- Ask questions about uploaded documents.
- Receive answers based on document content.
- Validate unsupported file types.
- Check backend health and connectivity.
- Display useful error messages when the backend is unavailable.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| Backend | Python, FastAPI |
| AI runtime | Ollama |
| PDF processing | pypdf |
| HTTP requests | Requests |
| API testing | FastAPI Swagger UI |

## Project Structure

```text
AI_PDF/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
├── .gitignore
└── README.md
```

## Run Locally

### Prerequisites

- Python installed
- Node.js and npm installed
- Ollama installed and running
- A compatible language model downloaded in Ollama

### 1. Start the backend

Open a terminal in the project directory:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

### 2. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local frontend URL printed by Vite in your terminal.

### 3. Start Ollama

Ensure Ollama is running and that the model configured in the backend is available locally. Check the model name and configuration in `backend/main.py`.

## Quality Assurance

Manual testing includes:

- Valid PDF upload
- Unsupported file type validation
- Questions with answers present in the PDF
- Questions about information missing from the PDF
- Empty-question validation
- Uploading a second PDF
- Backend health checks
- Error handling when the backend is unavailable

## Project Status

The project is maintained as a local development application. Deployment is not currently included.

## Future Improvements

- Expand automated test coverage.
- Improve the user interface and accessibility.
- Add structured test reports and bug tracking.
- Improve document processing and error handling.