import streamlit as st
import os
import sys
import time
from agents.collector import PaperCollector
from agents.generator import TopicGenerator
from agents.evaluator import TopicEvaluator
from utils.report_generator import generate_html_report
import config
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Config
st.set_page_config(
    page_title="Research Topics Brainstorming Tool",
    page_icon="🤖",
    layout="wide"
)

# Sidebar Configuration
st.sidebar.title("Configuration")
st.sidebar.header("Ollama Settings")
ollama_url = st.sidebar.text_input("Ollama URL", value=config.OLLAMA_BASE_URL)
model_embedding = st.sidebar.text_input("Embedding Model", value=config.MODEL_EMBEDDING)
model_generator = st.sidebar.text_input("Generator Model", value=config.MODEL_GENERATOR)
model_evaluator = st.sidebar.text_input("Evaluator Model", value=config.MODEL_EVALUATOR)

# Update config based on sidebar inputs (runtime only)
config.OLLAMA_BASE_URL = ollama_url
config.MODEL_EMBEDDING = model_embedding
config.MODEL_GENERATOR = model_generator
config.MODEL_EVALUATOR = model_evaluator

# Tabs
tab1, tab2 = st.tabs(["📚 Research & Brainstorm", "💬 RAG Chat"])

# --- Tab 1: Research & Brainstorm ---
with tab1:
    st.header("Research Topics Brainstorming Tool")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        keyword = st.text_input("Enter Research Keyword", placeholder="e.g., Artificial Intelligence in Healthcare")
    with col2:
        paper_limit = st.number_input("Paper Limit", min_value=10, max_value=500, value=config.DEFAULT_PAPER_LIMIT)
        topic_count = st.number_input("Topic Count", min_value=1, max_value=20, value=config.DEFAULT_TOPIC_COUNT)

    if st.button("Start Research", type="primary"):
        if not keyword:
            st.error("Please enter a keyword.")
        else:
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Step 1: Collect Papers
                status_container.info("Step 1: Collecting Papers & Creating Vector DB...")
                collector = PaperCollector()
                papers = collector.fetch_papers(keyword, limit=paper_limit)
                
                if not papers:
                    st.error("No papers found.")
                else:
                    st.success(f"Found {len(papers)} papers.")
                    vector_db = collector.create_vector_db(papers)
                    progress_bar.progress(30)
                    
                    if vector_db:
                        # Step 2: Generate Topics
                        status_container.info("Step 2: Generating Research Topics...")
                        generator = TopicGenerator()
                        topics = generator.generate_topics(vector_db, keyword, num_topics=topic_count)
                        progress_bar.progress(60)
                        
                        if topics:
                            # Step 3: Evaluate Topics
                            status_container.info("Step 3: Evaluating Topics...")
                            evaluator = TopicEvaluator()
                            evaluated_topics = evaluator.evaluate_topics(topics)
                            progress_bar.progress(80)
                            
                            # Step 4: Generate Report
                            status_container.info("Step 4: Generating Reports...")
                            os.makedirs(config.OUTPUT_REPORT_DIR, exist_ok=True)
                            
                            # English Report
                            filename_en = os.path.join(config.OUTPUT_REPORT_DIR, f"report_{keyword.replace(' ', '_')}.html")
                            generate_html_report(evaluated_topics, filename=filename_en)
                            
                            # Korean Report (Try/Except)
                            try:
                                from agents.translator import TopicTranslator
                                translator = TopicTranslator()
                                translated_topics = translator.translate_topics(evaluated_topics, target_language="Korean")
                                filename_ko = os.path.join(config.OUTPUT_REPORT_DIR, f"report_{keyword.replace(' ', '_')}_ko.html")
                                generate_html_report(translated_topics, filename=filename_ko)
                                st.success("Korean report generated successfully.")
                            except Exception as e:
                                st.warning(f"Korean translation failed: {e}")
                                filename_ko = None

                            progress_bar.progress(100)
                            status_container.success("Research Completed!")
                            
                            # Display Results
                            st.subheader("Generated Topics")
                            for item in evaluated_topics:
                                topic = item.topic
                                eval_result = item.evaluation
                                with st.expander(f"{topic.title} (Score: {eval_result.total_score})"):
                                    st.write(f"**Description:** {topic.background}")
                                    st.write(f"**Necessity:** {topic.necessity}")
                                    st.write(f"**Expected Effects:** {topic.expected_effects}")
                                    
                                    st.write("**Evaluation:**")
                                    st.write(f"- Originality: {eval_result.originality_score}")
                                    st.write(f"- Feasibility: {eval_result.feasibility_score}")
                                    st.write(f"- Impact: {eval_result.impact_score}")
                                    st.write(f"- Reasoning: {eval_result.reasoning}")

                                    st.write("**Related Papers:**")
                                    for paper in topic.related_papers:
                                        st.write(f"- [{paper.get('title', 'Unknown')}]({paper.get('url', '#')}) ({paper.get('year', 'N/A')})")
                            
                            # Download Buttons
                            with open(filename_en, "r") as f:
                                st.download_button("Download English Report", f, file_name=os.path.basename(filename_en))
                            
                            if filename_ko and os.path.exists(filename_ko):
                                with open(filename_ko, "r") as f:
                                    st.download_button("Download Korean Report", f, file_name=os.path.basename(filename_ko))
                                    
                        else:
                            st.error("Failed to generate topics.")
                    else:
                        st.error("Failed to create Vector DB.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Tab 2: RAG Chat ---
with tab2:
    st.header("Chat with Research articles Data")
    
    if not os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
        st.warning("Vector DB not found. Please run a research session first.")
    else:
        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat Messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about the collected papers..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Initialize RAG components (lazy loading)
                    embeddings = OllamaEmbeddings(
                        base_url=config.OLLAMA_BASE_URL,
                        model=config.MODEL_EMBEDDING
                    )
                    
                    vector_db = Chroma(
                        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
                        embedding_function=embeddings,
                        collection_name=config.COLLECTION_NAME
                    )
                    
                    llm = ChatOllama(
                        base_url=config.OLLAMA_BASE_URL,
                        model=config.MODEL_GENERATOR,
                        temperature=config.GENERATOR_TEMPERATURE
                    )
                    
                    template = """Answer the question based only on the following context:

                    {context}

                    Question: {question}

                    Answer:"""
                    
                    rag_prompt = ChatPromptTemplate.from_template(template)
                    
                    retriever = vector_db.as_retriever(search_kwargs={"k": config.VECTOR_DB_SEARCH_K})
                    
                    def format_docs(docs):
                        return "\n\n".join([d.page_content for d in docs])

                    chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | rag_prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Stream response
                    for chunk in chain.stream(prompt):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    full_response = f"Error: {str(e)}"
                    message_placeholder.error(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
