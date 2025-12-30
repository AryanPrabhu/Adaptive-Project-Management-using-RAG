# Project Assistant 🤖

An intelligent AI assistant that answers questions about your project using both documentation and live GitHub repository data.

## Features ✨

- **📄 PDF Documentation Search**: Semantic search through project documentation
- **🔗 GitHub Repository Analysis**: Live data from issues, PRs, commits, contributors, and releases
- **🤖 Intelligent Routing**: Hybrid keyword + AI-based routing to find the right data source
- **💬 Interactive Chat**: Both CLI and web-based Streamlit UI
- **🎯 Smart Caching**: Automatically detects when to refresh data vs. use cached databases

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Question                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Hybrid Router           │
        │  (Keywords + LLM)         │
        └──────┬──────────┬─────────┘
               │          │
      ┌────────▼──┐   ┌──▼────────┐
      │   PDF DB  │   │ GitHub DB │
      │  (FAISS)  │   │  (FAISS)  │
      └────────┬──┘   └──┬────────┘
               │          │
        ┌──────▼──────────▼─────┐
        │   LLM (Llama 3.2)     │
        │   Answer Generation   │
        └───────────┬───────────┘
                    │
            ┌───────▼───────┐
            │   Response    │
            └───────────────┘
```

## Setup 🚀

### Prerequisites

- Python 3.8+
- Ollama installed ([Download](https://ollama.ai))
- GitHub Personal Access Token

### Installation

1. **Clone or create the project directory**

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies**
```bash
pip install PyPDF2 sentence-transformers faiss-cpu numpy
pip install PyGithub
pip install langchain langchain-community ollama
pip install streamlit
```

4. **Install Ollama model**
```bash
ollama pull llama3.2:1b
```

### Configuration

#### For PDF Documentation

Edit `pdfinjest.py` and set your PDF path:
```python
pdf_path = '/path/to/your/document.pdf'
```

#### For GitHub Repository

Edit `extractgitinfo.py` and configure:
```python
GITHUB_TOKEN = "your_github_token_here"
REPO_OWNER = "owner_name"
REPO_NAME = "repo_name"
```

## Usage 📖

### Initial Data Ingestion

1. **Ingest PDF documentation**
```bash
python pdfinjest.py
```
This creates:
- `faiss_index.bin` - PDF vector database
- `metadata.pkl` - PDF metadata
- `pdf_config.json` - Tracks which PDF was indexed

2. **Ingest GitHub data**
```bash
python extractgitinfo.py
```
This creates:
- `github_faiss_index.bin` - GitHub vector database
- `github_metadata.pkl` - GitHub metadata
- `github_raw_data.json` - Raw GitHub API data
- `github_config.json` - Tracks which repo was indexed

### Running the Assistant

#### CLI Mode
```bash
python agent.py
```

#### Web UI (Streamlit)
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## Smart Caching 🧠

The system automatically detects when data needs to be refreshed:

### PDF Documents
- **Cached**: If the same PDF path is used, existing database is reused
- **Refreshed**: When PDF path changes or `pdf_config.json` is deleted

### GitHub Repository
- **Cached**: If same repo owner, name, and token are used
- **Refreshed**: When repo changes or `github_config.json` is deleted

### Manual Refresh

To force a refresh:
```bash
# Delete config files
rm pdf_config.json github_config.json

# Re-run ingestion
python pdfinjest.py
python extractgitinfo.py
```

Or in Streamlit UI, click the "Refresh Data Sources" button.

## How It Works 🔧

### 1. Data Ingestion

**PDF Processing:**
- Extracts text from PDF pages
- Splits into chunks (default: 500 characters)
- Creates embeddings using Sentence Transformers
- Stores in FAISS vector database

**GitHub Processing:**
- Fetches via GitHub API: issues, PRs, commits, contributors, releases
- Organizes data by type
- Creates embeddings for each item
- Stores in separate FAISS vector database

### 2. Question Routing

**Hybrid Approach:**
1. **Keyword Analysis**: Checks for specific terms (e.g., "open issues", "features")
2. **LLM Analysis**: Uses metadata about what's in each database to make intelligent routing
3. **Agreement Check**: 
   - If both agree → Route to that source
   - If keyword analysis is confident about GitHub → Use it
   - If disagreement → Ask user for clarification

### 3. Answer Generation

1. Retrieve top-k relevant chunks from appropriate database
2. Provide context to LLM (Llama 3.2)
3. Generate natural language answer
4. Include sources and similarity scores

## Example Questions 💡

### Documentation Questions
- "What is this project about?"
- "What are the main features?"
- "What technology stack is used?"
- "What are the system requirements?"

### GitHub Questions
- "How many open issues are there?"
- "What are the closed issues?"
- "Who are the contributors?"
- "What was the last commit?"
- "Are there any open pull requests?"

## File Structure 📁

```
communication/
├── pdfinjest.py              # PDF ingestion script
├── extractgitinfo.py         # GitHub data extraction script
├── agent.py                  # CLI agent
├── app.py                    # Streamlit web UI
├── faiss_index.bin           # PDF vector database
├── metadata.pkl              # PDF metadata
├── pdf_config.json           # PDF configuration
├── github_faiss_index.bin    # GitHub vector database
├── github_metadata.pkl       # GitHub metadata
├── github_config.json        # GitHub configuration
└── github_raw_data.json      # Raw GitHub API data
```

## Troubleshooting 🔍

### "Ollama not found"
Install Ollama from https://ollama.ai

### "Model not found"
```bash
ollama pull llama3.2:1b
```

### "Import errors"
Make sure you're in the virtual environment:
```bash
source venv/bin/activate
```

### "GitHub API rate limit"
Use a GitHub Personal Access Token with appropriate permissions.

### Force data refresh
Delete the config files:
```bash
rm pdf_config.json github_config.json
```

## Technologies Used 🛠️

- **LangChain**: LLM orchestration framework
- **Sentence Transformers**: Text embeddings (all-MiniLM-L6-v2)
- **FAISS**: Vector similarity search
- **Ollama**: Local LLM inference (Llama 3.2 1B)
- **PyGithub**: GitHub API client
- **Streamlit**: Web UI framework
- **PyPDF2**: PDF text extraction

## Performance ⚡

- **Model**: Llama 3.2 1B (lightweight, fast)
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Search**: FAISS L2 distance
- **Response Time**: 2-5 seconds per query

## License 📄

[Your License Here]

## Contributing 🤝

[Your Contributing Guidelines Here]
