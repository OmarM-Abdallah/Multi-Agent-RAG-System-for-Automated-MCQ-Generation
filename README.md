# RAG-Based Question Generation System

A multi-agent system for generating high-quality Multiple Choice Questions (MCQs) from PDF documents using Retrieval Augmented Generation (RAG).

## Architecture

### Multi-Agent System
- **Generator Agent**: Creates MCQ questions from retrieved context using Groq LLM
- **Evaluator Agent**: Scores and validates question quality (0-10 scale)
- **Quality Threshold**: Only questions scoring ≥ 7.0 are returned

### Tech Stack
- **Framework**: FastAPI
- **LLM Provider**: Groq (llama-3.3-70b-versatile)
- **Vector Database**: ChromaDB (persistent storage)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) - local, free
- **PDF Processing**: PyPDF

## Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Groq API key (free, no credit card required)

## Installation & Setup

### 1. Get Groq API Key

1. Visit https://console.groq.com/keys
2. Sign up (free, no credit card needed)
3. Create API key
4. Copy the key (starts with `gsk_...`)

### 2. Setup Project
```bash
# Clone/extract the project
cd rag-question-generator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run Application
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Note on Startup Time:** The first startup takes 2-3 minutes while the embedding model (80MB) loads into memory. This is a one-time initialization. Wait for the ```INFO: Started server process``` message before accessing the API.

### 4. Access API

Open your browser: http://localhost:8000/docs

## API Documentation

### Available Endpoints

#### `GET /health`
Health check endpoint
```bash
curl http://localhost:8000/health
```

#### `POST /ingest`
Upload and process PDF file
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@your_document.pdf"
```

**Returns:**
- Table of Contents
- Number of chunks created
- Confirmation message

#### `POST /generate/questions`
Generate MCQ questions based on query
```bash
curl -X POST "http://localhost:8000/generate/questions" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your topic here",
    "num_questions": 5
  }'
```

**Returns:**
- List of MCQ questions
- Evaluation scores (from Evaluator Agent)
- Retrieved context chunks

**Processing Time:** ~20-30 seconds (RAG retrieval + Generator + Evaluator)

#### `GET /stats`
Get vector database statistics
```bash
curl http://localhost:8000/stats
```

## Usage Examples

### Complete Workflow

**1. Start the application:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# Wait for "Application startup complete"
```

**2. Ingest a PDF:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@A_Quick_Algebra_Review.pdf"
```



**3. Generate questions:**
```bash
curl -X POST "http://localhost:8000/generate/questions" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "solving equations",
    "num_questions": 5
  }'
```

**NOTE:** You can use the web interface instead by visiting the url http://localhost:8000/docs


See `sample_questions.json` for complete example output.

## Docker Deployment

### Build and Run
```bash
# Build image
docker build -t rag-question-generator .

# Run container
docker run -d \
  --name rag-app \
  -p 8000:8000 \
  -e GROQ_API_KEY=your_groq_api_key_here \
  -e LLM_MODEL=llama-3.3-70b-versatile \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/vectordb:/app/vectordb \
  rag-question-generator

# View logs
docker logs -f rag-app

# Stop container
docker stop rag-app
docker rm rag-app
```

## Project Structure
```
rag-question-generator/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration settings
│   ├── agents/
│   │   ├── generator.py        # Question Generator Agent
│   │   └── evaluator.py        # Question Evaluator Agent
│   ├── services/
│   │   ├── pdf_processor.py    # PDF handling
│   │   ├── vector_store.py     # ChromaDB operations
│   │   └── rag_pipeline.py     # RAG orchestration
│   └── models/
│       └── schemas.py          # Pydantic models
├── tests/
│   └── test_api.py             # API tests
├── data/                       # Temporary PDF storage
├── vectordb/                   # ChromaDB persistence
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
└── sample_questions.json       # Example output
```

## Configuration

Environment variables (set in `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (required) | - |
| `LLM_MODEL` | Groq model name | llama-3.3-70b-versatile |
| `EMBEDDING_MODEL` | Local embedding model | all-MiniLM-L6-v2 |
| `CHUNK_SIZE` | Text chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap | 200 |
| `RETRIEVAL_TOP_K` | Chunks to retrieve | 5 |

## Testing

### Run Tests
```bash
pytest tests/test_api.py -v
```

### Test Results


- ✅ Application startup - All services initialized
- ✅ Health check endpoint - Returns 200 OK
- ✅ PDF ingestion - 31 chunks created from "A Quick Algebra Review.pdf"
- ✅ Table of Contents extraction - 5 sections identified
- ✅ Question generation - 5 questions generated with query "solving equations"
- ✅ Multi-agent workflow - Generator + Evaluator agents working
- ✅ Evaluation scoring - Questions scored 7.5-9.0, all approved

**Sample Output:** See `sample_questions.json`

## How It Works

1. **PDF Ingestion**
   - Extract text and structure from PDF
   - Split into semantic chunks (~1000 chars)
   - Generate embeddings using sentence-transformers
   - Store in ChromaDB vector database

2. **Question Generation**
   - User provides query/topic
   - RAG retrieval: Semantic search finds relevant chunks
   - Generator Agent: Creates MCQ questions from context
   - Evaluator Agent: Scores each question (0-10)
   - System filters: Returns only questions scoring ≥ 7.0



## Troubleshooting

**Issue: "No documents in vector store"**
- Run `/ingest` endpoint first to upload a PDF

**Issue: Port 8000 already in use**
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9
# Or use different port
python -m uvicorn app.main:app --port 8001
```

**Issue: Groq API error**
- Verify API key in `.env` file
- Check https://console.groq.com for available models
