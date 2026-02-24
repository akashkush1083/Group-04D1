"""
Education Creator Web Interface
- Upload PDFs and files
- Ask questions about PDF content
- Generate images
- Vector database storage
"""

import streamlit as st
import os
import json
from datetime import datetime
from education_creator import (
    SimpleVectorDB, 
    extract_text_from_pdf, 
    process_pdf_content, 
    answer_question_from_pdfs,
    generate_image_huggingface,
    generate_text,
    display_content
)
from io import BytesIO
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="Education Creator with PDF Q&A",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = SimpleVectorDB()
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Education Creator with PDF Q&A</h1>
        <p>Upload PDFs • Ask Questions • Generate Images • Store in Vector Database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload PDF", 
        "❓ Ask Questions", 
        "🎨 Generate Images", 
        "🔍 Search Topics", 
        "📊 Database"
    ])
    
    with tab1:
        st.header("📁 Upload PDF Files")
        st.write("Upload PDF files to extract content and store in the vector database.")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file to extract and process its content"
        )
        
        if uploaded_file is not None:
            st.info(f"File selected: {uploaded_file.name}")
            
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Extract text from PDF
                        pdf_text = extract_text_from_pdf(uploaded_file)
                        
                        if not pdf_text.strip():
                            st.error("❌ No text found in PDF!")
                            return
                        
                        st.success(f"✅ Extracted {len(pdf_text)} characters from PDF")
                        
                        # Process the content
                        with st.spinner("Analyzing content with AI..."):
                            content = process_pdf_content(pdf_text, uploaded_file.name)
                            topic = content['title']
                        
                        # Check for duplicates
                        duplicates = st.session_state.db.find_duplicates(topic)
                        if duplicates:
                            st.warning("⚠️ Similar topics found:")
                            for dup in duplicates:
                                st.write(f"• {dup['topic']} ({dup['similarity']*100:.1f}% similar)")
                            
                            if not st.checkbox("Continue anyway"):
                                return
                        
                        # Display content
                        st.subheader(f"📚 {topic}")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📖 Explanation:**")
                            st.write(content["explanation"])
                            
                            st.write("**🌍 Real-Life Example:**")
                            st.write(content["realLifeExample"])
                        
                        with col2:
                            st.write("**🪜 Step-by-Step:**")
                            for i, step in enumerate(content["stepByStep"], 1):
                                st.write(f"{i}. {step}")
                            
                            st.write("**💡 Key Points:**")
                            for point in content["keyPoints"]:
                                st.write(f"• {point}")
                        
                        st.write("**📋 Summary:**")
                        st.write(content["summary"])
                        
                        st.write("**🎴 Flashcard:**")
                        col_q, col_a = st.columns(2)
                        with col_q:
                            st.info(f"**Q:** {content['flashcard']['question']}")
                        with col_a:
                            st.success(f"**A:** {content['flashcard']['answer']}")
                        
                        # Add to database
                        st.session_state.db.add(topic, content)
                        
                        # Save to file
                        json_filename = f"{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(json_filename, 'w') as f:
                            json.dump({
                                "topic": topic,
                                "timestamp": datetime.now().isoformat(),
                                "content": content,
                                "source": "PDF",
                                "source_file": uploaded_file.name
                            }, f, indent=2)
                        
                        st.success(f"✅ Content saved to database and file: {json_filename}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
    
    with tab2:
        st.header("❓ Ask Questions About PDFs")
        st.write("Ask questions about the content of your uploaded PDF files.")
        
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What is photosynthesis? Explain the main concepts...",
            help="Ask any question about the content in your uploaded PDFs"
        )
        
        if st.button("🔍 Get Answer", type="primary") and question:
            with st.spinner("Searching PDFs and generating answer..."):
                try:
                    answer = answer_question_from_pdfs(question, st.session_state.db)
                    st.subheader("💡 Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab3:
        st.header("🎨 Generate Educational Images")
        st.write("Generate images automatically based on topics.")
        
        topic = st.text_input("Topic:", placeholder="e.g., Photosynthesis, Database Management, Random Forest")
        
        if st.button("🎨 Generate Image", type="primary") and topic:
            with st.spinner("Generating image..."):
                try:
                    # Automatically create a descriptive prompt from the topic
                    auto_prompt = f"Educational diagram of {topic}, clean minimalist style, white background, professional infographic, clear and simple illustration"
                    
                    filename = generate_image_huggingface(auto_prompt, topic)
                    
                    # Display the image
                    if os.path.exists(filename):
                        st.image(filename, caption=f"Generated image for: {topic}")
                        st.success(f"✅ Image saved: {filename}")
                    else:
                        st.error("❌ Image generation failed")
                except Exception as e:
                    st.error(f"❌ Error generating image: {str(e)}")
    
    with tab4:
        st.header("🔍 Search Topics")
        st.write("Search for similar topics in the database.")
        
        search_query = st.text_input(
            "Search query:",
            placeholder="e.g., plant biology, mathematics, chemistry...",
            help="Search for topics using natural language"
        )
        
        if st.button("🔍 Search", type="primary") and search_query:
            with st.spinner("Searching..."):
                results = st.session_state.db.search(search_query, top_k=10)
                
                if not results:
                    st.info("❌ No results found")
                else:
                    st.subheader(f"📋 Found {len(results)} Results")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"{i}. {result['topic']} ({result['similarity']*100:.1f}% similar)"):
                            st.write("**Summary:**")
                            st.write(result['content'].get('summary', 'N/A'))
                            
                            st.write("**Explanation:**")
                            st.write(result['content'].get('explanation', 'N/A'))
                            
                            if 'source_file' in result['content']:
                                st.info(f"📁 Source: {result['content']['source_file']}")
    
    with tab5:
        st.header("📊 Database Management")
        
        # Statistics
        stats = st.session_state.db.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Topics", stats['total_entries'])
        with col2:
            st.metric("Latest Topic", stats['latest'] or "None")
        with col3:
            if stats['topics']:
                st.metric("Unique Topics", len(set(stats['topics'])))
        
        # All topics
        if stats['topics']:
            st.subheader("📋 All Topics in Database")
            
            search_term = st.text_input("Filter topics:", placeholder="Type to filter...")
            
            filtered_topics = stats['topics']
            if search_term:
                filtered_topics = [t for t in stats['topics'] if search_term.lower() in t.lower()]
            
            for i, topic in enumerate(filtered_topics, 1):
                st.write(f"{i}. {topic}")
        
        # Database actions
        st.subheader("🔧 Database Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload Database", help="Reload database from file"):
                st.session_state.db = SimpleVectorDB()
                st.rerun()
        
        with col2:
            if st.button("📊 Show Detailed Stats"):
                st.json(stats)

if __name__ == "__main__":
    main()
