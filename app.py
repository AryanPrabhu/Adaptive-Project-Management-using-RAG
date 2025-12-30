import streamlit as st
import os
import sys
from agent import ProjectAgent
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Project Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-github {
        background-color: #e1f5ff;
        color: #01579b;
    }
    .badge-documentation {
        background-color: #e8f5e9;
        color: #1b5e20;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    """Load the agent (cached to avoid reloading)"""
    with st.spinner("🔄 Initializing Project Agent..."):
        agent = ProjectAgent(
            ollama_model="llama3.2:1b",
            run_ingestion=None  # Auto-detect: will create indexes if missing
        )
    return agent


def display_answer(result: Dict[str, Any]):
    """Display the answer with formatting"""
    category = result.get('category', 'unknown')
    answer = result.get('answer', 'No answer available')
    sources = result.get('sources', [])
    
    # Display category badge
    if category == 'github':
        st.markdown('<div class="category-badge badge-github">📊 GitHub Data</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="category-badge badge-documentation">📚 Documentation</div>', unsafe_allow_html=True)
    
    # Display answer
    st.markdown("### 📝 Answer")
    st.write(answer)
    
    # Display sources if available
    if sources:
        st.markdown("---")
        st.markdown(f"### 📚 Sources ({len(sources)} relevant items)")
        
        with st.expander("View Sources", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**Source {i}:**")
                st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                st.text(source.get('content', 'No content'))
                
                # Display metadata
                source_type = source.get('type', 'unknown')
                similarity = source.get('similarity_score', 0)
                st.caption(f"Type: {source_type} | Similarity: {similarity:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">🤖 Project Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your project documentation and GitHub repository</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/chatbot.png", width=100)
        st.title("About")
        st.info(
            """
            This AI assistant can answer questions from:
            
            **📄 Project Documentation**
            - Project features
            - Requirements
            - Architecture
            - Technology stack
            
            **🔗 GitHub Repository**
            - Open/closed issues
            - Pull requests
            - Contributors
            - Commits & releases
            """
        )
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        show_routing = st.checkbox("Show routing details", value=False)
        
        # Data refresh
        st.markdown("---")
        st.subheader("🔄 Data Management")
        
        if st.button("Refresh Data Sources"):
            with st.spinner("Refreshing data..."):
                # Clear cache and reload
                st.cache_resource.clear()
                st.success("Data refreshed! Please reload the page.")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 Statistics")
        
        try:
            agent = st.session_state.get('agent')
            if agent:
                if agent.pdf_retriever:
                    pdf_count = len(agent.pdf_retriever.doc_metadata)
                    st.metric("PDF Chunks", pdf_count)
                
                if agent.github_retriever:
                    gh_count = len(agent.github_retriever.doc_metadata)
                    st.metric("GitHub Chunks", gh_count)
        except:
            pass
    
    # Main content
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = load_agent()
        st.success("✅ Agent initialized successfully!")
    
    agent = st.session_state.agent
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_answer(message["content"])
            else:
                st.write(message["content"])
    
    # Example questions
    if not st.session_state.messages:
        st.markdown("### 💡 Example Questions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📄 Documentation:**")
            if st.button("What is this project about?"):
                st.session_state.current_question = "What is this project about?"
                st.rerun()
            if st.button("What are the main features?"):
                st.session_state.current_question = "What are the main features?"
                st.rerun()
            if st.button("What technology stack is used?"):
                st.session_state.current_question = "What technology stack is used?"
                st.rerun()
        
        with col2:
            st.markdown("**🔗 GitHub:**")
            if st.button("How many open issues are there?"):
                st.session_state.current_question = "How many open issues are there?"
                st.rerun()
            if st.button("Who are the contributors?"):
                st.session_state.current_question = "Who are the contributors?"
                st.rerun()
            if st.button("What was the last commit?"):
                st.session_state.current_question = "What was the last commit?"
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the project..."):
        st.session_state.current_question = prompt
    
    # Process question
    if 'current_question' in st.session_state and st.session_state.current_question:
        question = st.session_state.current_question
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.write(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Show routing if enabled
                if show_routing:
                    keyword_decision = agent._keyword_route(question)
                    llm_decision = agent._llm_route(question)
                    
                    st.caption(f"🔍 Keyword analysis: {keyword_decision}")
                    st.caption(f"🤖 AI analysis: {llm_decision}")
                
                # Get answer
                result = agent.ask(question)
                
                # Display answer
                display_answer(result)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": result})
        
        # Clear current question
        del st.session_state.current_question
    
    # Clear chat button
    if st.session_state.messages:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
