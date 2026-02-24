# Education Creator with PDF Q&A and Vector Database

An intelligent educational platform that processes PDFs, generates content, and answers questions using AI and vector similarity search.

## ✨ Features

- 📁 **PDF Upload & Processing**: Upload PDF files and extract text content automatically
- ❓ **PDF Q&A**: Ask questions about uploaded PDF content and get intelligent answers
- 🤖 **AI Content Generation**: Uses Google Gemini to generate comprehensive educational content
- 🔍 **Smart Search**: Vector-based similarity search to find related topics across PDFs and generated content
- 🚫 **Duplicate Detection**: Automatically detects similar existing topics
- 🎨 **Image Generation**: Creates educational images using HuggingFace models
- 💾 **Persistent Storage**: Saves all content to local vector database and JSON files
- 🌐 **Web Interface**: Modern Streamlit web app for easy file upload and interaction

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get API Keys

**Gemini API (Required)**
- Visit: https://makersuite.google.com/app/apikey
- Create a new API key
- Used for text generation, embeddings, and Q&A

**HuggingFace API (Optional for images)**
- Visit: https://huggingface.co/settings/tokens
- Create a READ token (free)
- Used for image generation

### 3. Configure Environment
```bash
# Copy the example file
copy .env.example .env

# Edit .env and add your API keys
# GEMINI_API_KEY=your_actual_gemini_key
# HF_TOKEN=your_actual_huggingface_token
```

### 4. Run the Application

**Option A: Web Interface (Recommended)**
```bash
streamlit run web_app.py
```
Then open http://localhost:8501 in your browser

**Option B: Command Line Interface**
```bash
python education_creator.py
```

## 📖 Usage Guide

### Web Interface Features

1. **📁 Upload PDF**: 
   - Upload PDF files through the web interface
   - Automatically extracts text and processes content
   - AI analyzes and structures the content
   - Stores in vector database for future searches

2. **❓ Ask Questions**:
   - Ask natural language questions about your uploaded PDFs
   - AI searches through all uploaded content
   - Provides accurate answers based on your documents

3. **🎨 Generate Images**:
   - Create educational images based on topics or PDF content
   - AI-powered image generation with multiple fallback options

4. **🔍 Search Topics**:
   - Search across all uploaded PDFs and generated content
   - Vector similarity search finds related content intelligently

5. **📊 Database Management**:
   - View statistics about your knowledge base
   - Browse all topics and documents
   - Manage your educational content

### Command Line Interface Features

1. **Learn New Topic**: Generate and store educational content
2. **Upload PDF File**: Process PDF files and add to database
3. **Search Similar Topics**: Find related content using vector search
4. **View All Topics**: Browse your knowledge database
5. **Check for Duplicates**: Avoid creating similar content
6. **Ask Question About PDFs**: Q&A functionality for uploaded documents
7. **Generate Images**: Create educational illustrations
8. **Database Statistics**: View your learning progress

## 🏗️ Project Structure

```
education_creator/
├── education_creator.py    # Main CLI application
├── web_app.py             # Streamlit web interface
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .env                   # Your API keys (create this)
├── education_vector_db.pkl # Vector database (auto-created)
└── README.md              # This file
```

## 🧠 How It Works

### PDF Processing
- Uses PyPDF2 to extract text from uploaded PDFs
- AI analyzes the extracted content and creates structured educational material
- Content is automatically stored in the vector database

### Vector Database
- Converts all text content into numerical embeddings using Gemini
- Enables semantic search and similarity matching
- Finds related content even with different wording

### Q&A System
- When you ask a question, it searches the vector database for relevant content
- AI uses the found context to generate accurate answers
- Works across all uploaded PDFs and generated content

### Image Generation
- Uses HuggingFace's free serverless inference
- Multiple model fallbacks ensure reliability
- Local fallback generation if online services fail

## 📝 Example Workflow

1. **Upload a PDF**: Upload your textbook or study materials
2. **Process Content**: AI extracts and structures the information
3. **Ask Questions**: "What is photosynthesis?" → Get detailed answers from your PDFs
4. **Generate Images**: Create visual aids for better understanding
5. **Search & Explore**: Find related topics across all your materials

## 🔧 Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key    # Required
HF_TOKEN=your_huggingface_token       # Optional (for images)
```

### Database Settings
- Database file: `education_vector_db.pkl`
- Auto-saves after each operation
- Can be reloaded at any time

## 🚨 Notes

- The vector database is created automatically on first run
- All content is stored locally for privacy
- PDF text extraction works best with text-based PDFs (not scanned images)
- Image generation requires internet connection for best results
- Local fallback images are created if online services fail

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.
