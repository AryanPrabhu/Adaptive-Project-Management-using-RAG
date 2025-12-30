import os
import subprocess
import sys
from typing import List, Dict, Any
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from extractgitinfo import GitHubVectorStore


class PDFRetriever:
    """Custom FAISS retriever for PDF document search"""
    
    def __init__(self, index_path: str, metadata_path: str, model_name: str = 'all-MiniLM-L6-v2', k: int = 5):
        self.embedding_model = SentenceTransformer(model_name)
        self.faiss_index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            self.doc_metadata = pickle.load(f)
        
        self.top_k = k
        print(f"✓ Loaded FAISS index with {self.faiss_index.ntotal} vectors from {index_path}")
    
    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on query"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), self.top_k)
        
        # Prepare documents
        documents = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.doc_metadata):
                metadata_item = self.doc_metadata[idx]
                doc = {
                    'content': metadata_item.get('text', ''),
                    'type': metadata_item.get('type', 'unknown'),
                    'distance': float(distances[0][i]),
                    'similarity_score': 1 / (1 + float(distances[0][i])),
                    'metadata': metadata_item.get('metadata', {})
                }
                documents.append(doc)
        
        return documents


class ProjectAgent:
    """
    LangChain-based agent that answers questions using:
    1. PDF documentation (from pdfinjest.py)
    2. GitHub repository data (from extractgitinfo.py)
    """
    
    def __init__(
        self,
        ollama_model: str = "llama3.2:1b",
        pdf_index_path: str = "faiss_index.bin",
        pdf_metadata_path: str = "metadata.pkl",
        github_index_path: str = "github_faiss_index.bin",
        github_metadata_path: str = "github_metadata.pkl",
        run_ingestion: bool = None  # None = auto-detect
    ):
        print("\n" + "="*80)
        print("INITIALIZING PROJECT AGENT")
        print("="*80 + "\n")
        
        # Step 1: Auto-detect if ingestion is needed or run if requested
        if run_ingestion is None:
            # Auto-detect: run ingestion if indexes don't exist
            pdf_exists = os.path.exists(pdf_index_path) and os.path.exists(pdf_metadata_path)
            github_exists = os.path.exists(github_index_path) and os.path.exists(github_metadata_path)
            
            if not pdf_exists or not github_exists:
                print("📥 Indexes not found. Running data ingestion...\n")
                self._run_ingestion_scripts()
            else:
                print("✓ Found existing indexes. Skipping ingestion.\n")
        elif run_ingestion:
            self._run_ingestion_scripts()
        
        # Step 2: Check if Ollama is installed and model is available
        self._setup_ollama(ollama_model)
        
        # Step 3: Load retrievers
        self.pdf_retriever = None
        self.github_retriever = None
        
        if os.path.exists(pdf_index_path) and os.path.exists(pdf_metadata_path):
            print("\nLoading PDF documentation retriever...")
            self.pdf_retriever = PDFRetriever(
                index_path=pdf_index_path,
                metadata_path=pdf_metadata_path,
                k=3
            )
        else:
            print(f"⚠ PDF index not found at {pdf_index_path}")
        
        if os.path.exists(github_index_path) and os.path.exists(github_metadata_path):
            print("\nLoading GitHub data retriever...")
            self.github_retriever = GitHubVectorStore()
            self.github_retriever.load_index(
                index_path=github_index_path,
                metadata_path=github_metadata_path
            )
        else:
            print(f"⚠ GitHub index not found at {github_index_path}")
        
        # Step 4: Initialize LLM
        print(f"\nInitializing Ollama model: {ollama_model}")
        self.llm = Ollama(model=ollama_model, temperature=0.7)
        print("✓ Ollama model initialized")
        
        # Step 5: Create chains
        self._create_chains()
        
        print("\n" + "="*80)
        print("✅ PROJECT AGENT READY")
        print("="*80 + "\n")
    
    def _run_ingestion_scripts(self):
        """Run PDF and GitHub ingestion scripts"""
        print("Running data ingestion scripts...\n")
        
        # Run PDF ingestion
        if os.path.exists("pdfinjest.py"):
            print("→ Running pdfinjest.py...")
            try:
                result = subprocess.run(
                    [sys.executable, "pdfinjest.py"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    print("✓ PDF ingestion completed\n")
                else:
                    print(f"⚠ PDF ingestion had issues: {result.stderr}\n")
            except Exception as e:
                print(f"⚠ Error running pdfinjest.py: {e}\n")
        
        # Run GitHub ingestion
        if os.path.exists("extractgitinfo.py"):
            print("→ Running extractgitinfo.py...")
            try:
                result = subprocess.run(
                    [sys.executable, "extractgitinfo.py"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    print("✓ GitHub ingestion completed\n")
                else:
                    print(f"⚠ GitHub ingestion had issues: {result.stderr}\n")
            except Exception as e:
                print(f"⚠ Error running extractgitinfo.py: {e}\n")
    
    def _setup_ollama(self, model_name: str):
        """Setup and verify Ollama installation"""
        print("Checking Ollama installation...")
        
        # Check if Ollama is installed
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✓ Ollama is installed: {result.stdout.strip()}")
            else:
                print("⚠ Ollama not found. Please install from https://ollama.ai")
                sys.exit(1)
        except FileNotFoundError:
            print("⚠ Ollama not found. Please install from https://ollama.ai")
            print("Installation: Run 'brew install ollama' or download from https://ollama.ai")
            sys.exit(1)
        
        # Check if model exists, if not pull it
        print(f"\nChecking for model: {model_name}")
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if model_name.split(':')[0] not in result.stdout:
                print(f"→ Pulling model {model_name} (this may take a few minutes)...")
                pull_result = subprocess.run(
                    ["ollama", "pull", model_name],
                    timeout=600
                )
                if pull_result.returncode == 0:
                    print(f"✓ Model {model_name} downloaded successfully")
                else:
                    print(f"⚠ Failed to download model {model_name}")
                    sys.exit(1)
            else:
                print(f"✓ Model {model_name} is available")
        except Exception as e:
            print(f"⚠ Error checking/pulling model: {e}")
    
    def _create_chains(self):
        """Create LangChain chains for different query types"""
        
        # Chain for documentation questions
        self.doc_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on project documentation.

Context from documentation:
{context}

Question: {question}

Answer: Provide a clear and concise answer based on the context provided. If the context doesn't contain enough information to answer the question, say so."""
        )
        
        self.doc_chain = LLMChain(llm=self.llm, prompt=self.doc_prompt)
        
        # Chain for GitHub status questions
        self.github_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions about a GitHub repository's status.

Context from GitHub repository:
{context}

Question: {question}

Answer: Provide a clear summary based on the GitHub data provided. Include relevant numbers, dates, and specific details."""
        )
        
        self.github_chain = LLMChain(llm=self.llm, prompt=self.github_prompt)
        
        # Chain for routing questions with metadata context
        self.router_prompt = PromptTemplate(
            input_variables=["question", "pdf_info", "github_info"],
            template="""You are a routing assistant. You have access to two data sources:

PDF DOCUMENTATION contains:
{pdf_info}

GITHUB REPOSITORY DATA contains:
{github_info}

Based on the question and what each data source contains, determine which source can answer it.

Question: {question}

If the question asks about:
- Project purpose, features, requirements, design, architecture, technology -> answer "documentation"
- Current issues, PRs, commits, contributors, repository statistics, activity -> answer "github"

Answer with ONLY ONE WORD - either "documentation" or "github":"""
        )
        
        self.router_chain = LLMChain(llm=self.llm, prompt=self.router_prompt)
    
    def _keyword_route(self, question: str) -> str:
        """Route based on keyword matching"""
        question_lower = question.lower()
        
        # GitHub-specific keywords
        github_keywords = [
            "how many issue", "number of issue", "total issue", "issue count",
            "how many pr", "number of pr", "total pr", "pull request count",
            "open issue", "closed issue", "issue #",
            "open pr", "closed pr", "merged pr", "pr #", "pull request #",
            "how many commit", "number of commit", "total commit", "commit count",
            "last commit", "recent commit", "latest commit",
            "how many contributor", "number of contributor", "who is contributor",
            "contributor", "contributors", "who is working", "active contributor",
            "how many branch", "number of branch", "branch count",
            "how many release", "number of release", "latest release",
            "bug", "bugs", "fixed", "fixes", "resolved",
            "what was fixed", "what issues", "which issues"
        ]
        
        # Documentation keywords
        doc_keywords = [
            "what is the project", "what is this project", "about the project",
            "describe the", "explain the", "overview of",
            "what are the feature", "what features", "main feature",
            "what technology", "tech stack", "architecture",
            "how does it work", "how to use", "requirements",
            "what is cinebook", "purpose of"
        ]
        
        # Check for specific PR/Issue number references (e.g., "#143", "pr #143", "issue #143")
        import re
        if re.search(r'(pr|pull request|issue)\s*#?\d+', question_lower) or re.search(r'#\d+', question_lower):
            return "github"
        
        # Check for bug-related queries
        if any(word in question_lower for word in ["bug", "bugs", "fixed", "fixes", "resolved"]):
            return "github"
        
        # Check GitHub keywords
        github_score = sum(1 for keyword in github_keywords if keyword in question_lower)
        
        # Check documentation keywords
        doc_score = sum(1 for keyword in doc_keywords if keyword in question_lower)
        
        if github_score > doc_score:
            return "github"
        elif doc_score > github_score:
            return "documentation"
        else:
            return "unclear"
    
    def _get_data_source_info(self):
        """Get metadata about what's in each data source"""
        pdf_info = "Unknown - PDF index not loaded"
        github_info = "Unknown - GitHub index not loaded"
        
        # Get PDF metadata info
        if self.pdf_retriever:
            types = {}
            for item in self.pdf_retriever.doc_metadata:
                item_type = item.get('type', 'unknown')
                types[item_type] = types.get(item_type, 0) + 1
            
            pdf_info = f"Project documentation with {len(self.pdf_retriever.doc_metadata)} chunks covering: "
            pdf_info += "project requirements, features, specifications, architecture, and design details."
        
        # Get GitHub metadata info
        if self.github_retriever:
            types = {}
            for item in self.github_retriever.doc_metadata:
                item_type = item.get('type', 'unknown')
                types[item_type] = types.get(item_type, 0) + 1
            
            type_summary = []
            if types.get('open_issue', 0) > 0:
                type_summary.append(f"{types['open_issue']} open issues")
            if types.get('closed_issue', 0) > 0:
                type_summary.append(f"{types['closed_issue']} closed issues")
            if types.get('open_pull_request', 0) > 0:
                type_summary.append(f"{types['open_pull_request']} open PRs")
            if types.get('closed_pull_request', 0) > 0:
                type_summary.append(f"{types['closed_pull_request']} closed PRs")
            if types.get('commit', 0) > 0:
                type_summary.append(f"{types['commit']} commits")
            if types.get('contributors_summary', 0) > 0:
                type_summary.append("contributor information")
            if types.get('branches_summary', 0) > 0:
                type_summary.append("branch information")
            if types.get('release', 0) > 0:
                type_summary.append(f"{types['release']} releases")
            
            github_info = f"Live repository data including: {', '.join(type_summary)}."
        
        return pdf_info, github_info
    
    def _llm_route(self, question: str) -> str:
        """Route based on LLM decision with metadata context"""
        try:
            # Get metadata about data sources
            pdf_info, github_info = self._get_data_source_info()
            
            # Use LLM with context about what's in each source
            result = self.router_chain.run(
                question=question,
                pdf_info=pdf_info,
                github_info=github_info
            )
            category = result.strip().lower()
            
            if "github" in category:
                return "github"
            elif "documentation" in category or "doc" in category:
                return "documentation"
            else:
                return "unclear"
        except Exception as e:
            print(f"⚠ LLM routing error: {e}")
            return "unclear"
    
    def _ask_user_for_clarification(self, question: str) -> str:
        """Ask user to clarify what type of information they need"""
        print("\n" + "="*80)
        print("🤔 I need clarification to answer your question accurately.")
        print("="*80)
        print("\nYour question could be answered from different sources:")
        print("\n1. PROJECT DOCUMENTATION - Information about:")
        print("   • What the project is and its purpose")
        print("   • Features, requirements, and specifications")
        print("   • Architecture and technology stack")
        print("   • How the system is designed to work")
        print("\n2. GITHUB REPOSITORY DATA - Live information about:")
        print("   • Current open/closed issues and pull requests")
        print("   • Contributors and commit history")
        print("   • Branches and releases")
        print("   • Project activity and status")
        print("\n" + "="*80)
        
        while True:
            choice = input("\nWhich would you like to search? (1 for Documentation / 2 for GitHub): ").strip()
            if choice == "1":
                print("✓ Will search PROJECT DOCUMENTATION\n")
                return "documentation"
            elif choice == "2":
                print("✓ Will search GITHUB REPOSITORY DATA\n")
                return "github"
            else:
                print("⚠ Please enter 1 or 2")
    
    def _route_question(self, question: str) -> str:
        """
        Hybrid routing: keyword matching + LLM decision with user clarification fallback
        """
        # Step 1: Keyword-based routing
        keyword_decision = self._keyword_route(question)
        
        # Step 2: LLM-based routing (with metadata context)
        llm_decision = self._llm_route(question)
        
        # Step 3: Compare decisions with improved logic
        if keyword_decision == llm_decision and keyword_decision != "unclear":
            # Both agree and are confident - use it!
            return keyword_decision
        
        elif keyword_decision != "unclear" and llm_decision != "unclear":
            # Both are confident but disagree
            # Trust keyword matching for very clear cases, otherwise ask
            if keyword_decision == "github":
                # Keywords strongly indicate GitHub, trust it
                return keyword_decision
            else:
                # Ask for clarification when there's disagreement
                print(f"\n🔍 Keyword analysis suggests: {keyword_decision}")
                print(f"🤖 AI analysis suggests: {llm_decision}")
                return self._ask_user_for_clarification(question)
        
        elif keyword_decision != "unclear":
            # Only keyword matching is confident
            return keyword_decision
        
        elif llm_decision != "unclear":
            # Only LLM is confident
            return llm_decision
        
        else:
            # Both are unclear - ask user
            print(f"\n🔍 Keyword analysis suggests: {keyword_decision}")
            print(f"🤖 AI analysis suggests: {llm_decision}")
            return self._ask_user_for_clarification(question)
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using appropriate data source
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        print("\n" + "-"*80)
        print(f"Question: {question}")
        print("-"*80)
        
        # Route the question
        category = self._route_question(question)
        print(f"Category: {category}")
        
        if category == "github" and self.github_retriever:
            return self._answer_github_question(question)
        elif category == "documentation" and self.pdf_retriever:
            return self._answer_documentation_question(question)
        else:
            return {
                "answer": "I don't have the necessary data loaded to answer this question.",
                "category": category,
                "sources": []
            }
    
    def _answer_documentation_question(self, question: str) -> Dict[str, Any]:
        """Answer questions from PDF documentation"""
        print("→ Searching documentation...")
        
        # Retrieve relevant documents
        docs = self.pdf_retriever.get_relevant_documents(question)
        
        if not docs:
            return {
                "answer": "I couldn't find relevant information in the documentation.",
                "category": "documentation",
                "sources": []
            }
        
        # Prepare context
        context = "\n\n".join([doc['content'] for doc in docs])
        
        # Generate answer
        print("→ Generating answer...")
        answer = self.doc_chain.run(context=context, question=question)
        
        # Prepare sources
        sources = []
        for doc in docs:
            sources.append({
                "content": doc['content'][:200] + "...",
                "type": doc['type'],
                "similarity_score": doc['similarity_score']
            })
        
        return {
            "answer": answer.strip(),
            "category": "documentation",
            "sources": sources
        }
    
    def _answer_github_question(self, question: str) -> Dict[str, Any]:
        """Answer questions from GitHub data"""
        print("→ Searching GitHub data...")
        
        # Retrieve relevant documents using the intelligent search
        docs = self.github_retriever.search(question, k=5)
        
        if not docs:
            return {
                "answer": "I couldn't find relevant information in the GitHub data.",
                "category": "github",
                "sources": []
            }
        
        # --- START: Context Refinement Logic ---
        # For summary queries, prioritize the summary chunk to prevent hallucination
        summary_keywords = ['total', 'count', 'how many', 'summarize', 'summary', 'number of']
        is_summary_query = any(keyword in question.lower() for keyword in summary_keywords)
        
        summary_doc = next((doc for doc in docs if doc.get('type') == 'repository_summary'), None)

        if is_summary_query and summary_doc:
            print("✓ Prioritizing repository_summary chunk for focused answer.")
            context_docs = [summary_doc]
        else:
            context_docs = docs
        # --- END: Context Refinement Logic ---

        # Prepare context from the refined list of documents
        context = "\n\n".join([doc['text'] for doc in context_docs])
        
        # Generate answer
        print("→ Generating answer...")
        answer = self.github_chain.run(context=context, question=question)
        
        # Prepare sources from the original retrieved documents for user visibility
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.get('text', '')[:200] + "...",
                "type": doc.get('type', 'unknown'),
                "similarity_score": doc.get('similarity_score', 0.0)
            })
        
        return {
            "answer": answer.strip(),
            "category": "github",
            "sources": sources
        }
    
    def interactive_mode(self):
        """Run agent in interactive Q&A mode"""
        print("\n" + "="*80)
        print("INTERACTIVE MODE - Ask questions about the project!")
        print("Commands: 'quit' or 'exit' to stop, 'help' for examples")
        print("="*80 + "\n")
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                # Get answer
                result = self.ask(question)
                
                # Display answer
                print("\n📝 Answer:")
                print(result['answer'])
                
                # Display sources
                if result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])} relevant items found)")
                
                print("\n" + "-"*80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        """Show example questions"""
        print("\n" + "="*80)
        print("EXAMPLE QUESTIONS")
        print("="*80)
        print("\nDocumentation Questions:")
        print("  - What is this project about?")
        print("  - What are the main features?")
        print("  - What technology stack is used?")
        print("  - What are the system requirements?")
        print("\nGitHub Status Questions:")
        print("  - What are the open issues?")
        print("  - How many contributors are working on this?")
        print("  - What is the project status?")
        print("  - What was the last commit?")
        print("  - Are there any open pull requests?")
        print("="*80)


# Main execution
if __name__ == "__main__":
    # Initialize the agent
    # run_ingestion options:
    #   None (default) = Auto-detect: runs ingestion if indexes are missing
    #   True = Always run ingestion (refresh data)
    #   False = Never run ingestion (use existing indexes only)
    agent = ProjectAgent(
        ollama_model="llama3.2:1b",  # Lightweight and fast model
        run_ingestion=None  # Auto-detect and create indexes if missing
    )
    
    # Run in interactive mode
    agent.interactive_mode()
