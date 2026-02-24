"""
Education Creator with Vector Database
- Stores all topics you learn
- Finds similar topics
- Avoids duplicates
- Smart search
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from urllib.parse import quote
import requests
import numpy as np
from typing import List, Dict
import pickle
import PyPDF2
from io import BytesIO
import base64

# Load environment
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found!")
    exit(1)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Get text model
TEXT_MODEL = None
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            TEXT_MODEL = m.name
            print(f"✅ Using: {TEXT_MODEL}")
            break
except:
    TEXT_MODEL = "models/gemini-pro"

if not TEXT_MODEL:
    print("❌ No text models available!")
    exit(1)


# ============================================
# VECTOR DATABASE CLASS
# ============================================

class SimpleVectorDB:
    """
    Simple Vector Database for Educational Content
    Uses Gemini embeddings to store and search topics
    """
    
    def __init__(self, db_file="education_vector_db.pkl"):
        """Initialize vector database"""
        self.db_file = db_file
        self.data = []  # List of {id, topic, content, vector, timestamp}
        self.load()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Convert text to vector using Gemini
        This is the MAGIC that makes similarity search work!
        """
        try:
            # Try the newer embedding model first
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"⚠️  Embedding generation failed: {e}")
            # Fallback: simple hash-based embedding
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str, dim: int = 128) -> List[float]:
        """
        Fallback: Create simple embedding without API
        (Not as good as Gemini, but works offline)
        """
        # Simple character-based embedding
        vector = [0.0] * dim
        for i, char in enumerate(text.lower()[:dim]):
            vector[i] = ord(char) / 255.0
        return vector
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate how similar two vectors are
        Returns: 0.0 (totally different) to 1.0 (identical)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add(self, topic: str, content: Dict):
        """
        Add new educational content to database
        
        Args:
            topic: Topic name (e.g., "Photosynthesis")
            content: Full educational content dict
        """
        print(f"\n📊 Generating embedding for '{topic}'...")
        
        # Create text for embedding (combine key fields)
        text_for_embedding = f"{topic}. {content.get('explanation', '')} {content.get('summary', '')}"
        
        # Generate vector
        vector = self.generate_embedding(text_for_embedding)
        
        # Create entry
        entry = {
            'id': len(self.data) + 1,
            'topic': topic,
            'content': content,
            'vector': vector,
            'timestamp': datetime.now().isoformat()
        }
        
        self.data.append(entry)
        self.save()
        
        print(f"✅ Added to database! Total entries: {len(self.data)}")
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict]:
        """
        Find similar topics in database
        
        Args:
            query: Search query (e.g., "plant biology")
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
        
        Returns:
            List of similar entries with similarity scores
        """
        if not self.data:
            return []
        
        print(f"\n🔍 Searching for: '{query}'...")
        
        # Generate query vector
        query_vector = self.generate_embedding(query)
        
        # Calculate similarity with all entries
        results = []
        for entry in self.data:
            similarity = self.cosine_similarity(query_vector, entry['vector'])
            # Only include results above minimum similarity threshold
            if similarity >= min_similarity:
                results.append({
                    'topic': entry['topic'],
                    'similarity': similarity,
                    'content': entry['content'],
                    'timestamp': entry['timestamp']
                })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def find_duplicates(self, topic: str, threshold: float = 0.85) -> List[Dict]:
        """
        Check if similar topic already exists
        
        Args:
            topic: Topic to check
            threshold: Similarity threshold (0.0 to 1.0)
        
        Returns:
            List of similar existing topics
        """
        results = self.search(topic, top_k=3)
        duplicates = [r for r in results if r['similarity'] >= threshold]
        return duplicates
    
    def get_all_topics(self) -> List[str]:
        """Get list of all topics in database"""
        return [entry['topic'] for entry in self.data]
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_entries': len(self.data),
            'topics': self.get_all_topics(),
            'latest': self.data[-1]['topic'] if self.data else None
        }
    
    def save(self):
        """Save database to file"""
        with open(self.db_file, 'wb') as f:
            pickle.dump(self.data, f)
    
    def load(self):
        """Load database from file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"✅ Loaded {len(self.data)} entries from database")
            except:
                print("⚠️  Could not load database, starting fresh")
                self.data = []
        else:
            print("📂 Creating new database")
            self.data = []


# ============================================
# CONTENT GENERATION
# ============================================

def generate_text(topic):
    """Generate educational content using Gemini"""
    
    prompt = f"""You are an expert educational content creator.

Topic: {topic}

Return ONLY valid JSON (no markdown, no code blocks):

{{
  "explanation": "5-7 lines simple explanation",
  "realLifeExample": "clear real-life example",
  "stepByStep": ["step 1", "step 2", "step 3"],
  "keyPoints": ["point 1", "point 2", "point 3"],
  "summary": "3 line summary",
  "flashcard": {{
    "question": "question",
    "answer": "answer"
  }},
  "imagePrompt": "simple clear prompt for image generation (keep it short)"
}}"""

    model = genai.GenerativeModel(TEXT_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 2048,
        }
    )
    
    text = response.text.strip()
    text = text.replace('```json', '').replace('```', '').strip()
    
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start == -1 or end == 0:
        raise ValueError("No JSON found in response")
    
    return json.loads(text[start:end])


def generate_clean_filename(topic: str, max_length: int = 30) -> str:
    """Generate a clean, short filename from topic"""
    import re
    
    # Remove special characters and extra spaces
    clean_topic = re.sub(r'[^\w\s-]', '', topic)
    clean_topic = re.sub(r'\s+', '_', clean_topic.strip())
    
    # Truncate if too long
    if len(clean_topic) > max_length:
        clean_topic = clean_topic[:max_length].rstrip('_')
    
    # If empty after cleaning, use a default
    if not clean_topic:
        clean_topic = "generated_image"
    
    return f"{clean_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"


def generate_image_huggingface(prompt, topic):
    """Generate image using HuggingFace FREE Serverless Inference"""
    from huggingface_hub import InferenceClient
    
    print("  🎨 Generating image with HuggingFace (FREE)...")
    
    try:
        # Get HF token from environment
        HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        
        if not HF_TOKEN:
            raise Exception("HF_TOKEN or HUGGINGFACE_API_KEY not found in environment!")
        
        # Initialize client WITHOUT provider (uses free serverless)
        client = InferenceClient(token=HF_TOKEN)
        
        # Enhance prompt for educational content
        enhanced_prompt = (
            f"{prompt}, educational diagram, clean minimalist style, "
            f"white background, simple illustration, professional infographic"
        )
        
        # Try multiple models in order - with your preferred model first
        models = [
            "stabilityai/stable-diffusion-xl-base-1.0",  # Your preferred model
            "black-forest-labs/FLUX.1-schnell",        # Fast, free, good quality
            "stabilityai/stable-diffusion-2-1",         # Classic, reliable
            "runwayml/stable-diffusion-v1-5",           # Very stable, always works
        ]
        
        last_error = None
        for model in models:
            try:
                print(f"  📡 Trying model: {model}")
                
                # Generate image (FREE serverless inference)
                image = client.text_to_image(
                    enhanced_prompt,
                    model=model,
                )
                
                # Save the image (output is a PIL.Image object)
                filename = generate_clean_filename(topic)
                image.save(filename)
                
                print(f"  ✅ Success with {model}!")
                return filename
                
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                
                # If model is loading (503), wait and retry
                if "503" in error_msg or "loading" in error_msg.lower():
                    print(f"  ⏳ Model loading, waiting 20s...")
                    import time
                    time.sleep(20)
                    
                    # Retry once
                    try:
                        image = client.text_to_image(enhanced_prompt, model=model)
                        filename = generate_clean_filename(topic)
                        image.save(filename)
                        print(f"  ✅ Success with {model} on retry!")
                        return filename
                    except:
                        pass
                
                print(f"  ⚠️  {model} failed: {error_msg[:80]}")
                continue
        
        # All models failed
        raise Exception(f"All models failed. Last error: {last_error}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ⚠️  HuggingFace generation failed: {error_msg}")
        
        # Fallback to local generation if HF fails
        print("  🔄 Falling back to local generation...")
        return generate_image_local_fallback(prompt, topic)


def generate_image_local_fallback(prompt, topic):
    """Fallback: Generate local educational card if HF fails"""
    from PIL import Image, ImageDraw, ImageFont
    
    print("  🎨 Generating LOCAL educational card...")
    
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw border
    draw.rectangle([10, 10, width-10, height-10], outline=(70, 130, 180), width=10)
    
    # Draw title
    title = topic.upper()
    try:
        title_font = ImageFont.truetype("arial.ttf", 40)
        body_font = ImageFont.truetype("arial.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Center title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_width) // 2, 60), title, fill=(0, 51, 102), font=title_font)
    
    # Draw prompt
    prompt_lines = [prompt[i:i+40] for i in range(0, len(prompt), 40)]
    y = 150
    for line in prompt_lines[:5]:
        bbox = draw.textbbox((0, 0), line, font=body_font)
        line_width = bbox[2] - bbox[0]
        draw.text(((width - line_width) // 2, y), line, fill=(25, 25, 25), font=body_font)
        y += 40
    
    # Footer
    draw.rectangle([0, height-100, width, height], fill=(70, 130, 180))
    footer = "Educational Content Generator"
    footer_bbox = draw.textbbox((0, 0), footer, font=body_font)
    footer_width = footer_bbox[2] - footer_bbox[0]
    draw.text(((width - footer_width) // 2, height-70), footer, fill='white', font=body_font)
    
    filename = generate_clean_filename(topic)
    img.save(filename)
    
    print("  ✅ Local educational card created!")
    return filename


def display_content(content, topic):
    """Display content"""
    print("\n" + "=" * 60)
    print(f"📚 TOPIC: {topic.upper()}")
    print("=" * 60)
    
    print("\n📖 EXPLANATION:")
    print(content["explanation"])
    
    print("\n🌍 REAL-LIFE EXAMPLE:")
    print(content["realLifeExample"])
    
    print("\n🪜 STEP-BY-STEP:")
    for i, step in enumerate(content["stepByStep"], 1):
        print(f"  {i}. {step}")
    
    print("\n💡 KEY POINTS:")
    for point in content["keyPoints"]:
        print(f"  • {point}")
    
    print("\n📋 SUMMARY:")
    print(content["summary"])
    
    print("\n🎴 FLASHCARD:")
    print(f"  Q: {content['flashcard']['question']}")
    print(f"  A: {content['flashcard']['answer']}")


# ============================================
# PDF PROCESSING FUNCTIONS
# ============================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def process_pdf_content(pdf_text: str, filename: str) -> Dict:
    """Process PDF text and create structured content"""
    
    # Take first 2000 characters for processing (to stay within limits)
    sample_text = pdf_text[:2000] if len(pdf_text) > 2000 else pdf_text
    
    prompt = f"""You are an expert educational content analyzer. 

Analyze this PDF content and create structured educational material:

PDF Filename: {filename}
Content: {sample_text}

Return ONLY valid JSON (no markdown, no code blocks):

{{
  "title": "brief descriptive title based on content",
  "explanation": "5-7 lines explanation of the main topic",
  "realLifeExample": "clear real-life example from the content",
  "stepByStep": ["step 1", "step 2", "step 3"],
  "keyPoints": ["point 1", "point 2", "point 3"],
  "summary": "3 line summary",
  "flashcard": {{
    "question": "important question from the content",
    "answer": "answer from the content"
  }},
  "imagePrompt": "simple clear prompt for image generation based on content"
}}"""

    model = genai.GenerativeModel(TEXT_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 2048,
        }
    )
    
    text = response.text.strip()
    text = text.replace('```json', '').replace('```', '').strip()
    
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start == -1 or end == 0:
        raise ValueError("No JSON found in response")
    
    content = json.loads(text[start:end])
    content['full_text'] = pdf_text  # Store full text for Q&A
    content['source_file'] = filename
    
    return content


def answer_question_from_pdfs(question: str, db) -> str:
    """Answer questions using stored PDF content"""
    
    # Search for relevant content
    results = db.search(question, top_k=5)
    
    if not results:
        return "❌ No relevant content found in uploaded PDFs."
    
    # Build context from search results
    context = ""
    pdf_found = False
    
    for i, result in enumerate(results, 1):
        content = result['content']
        
        # Check if this is from a PDF
        if 'source_file' in content or 'full_text' in content:
            pdf_found = True
            context += f"\n--- Document {i}: {result['topic']} ---\n"
            context += f"Source File: {content.get('source_file', 'Unknown')}\n"
            context += f"Summary: {content.get('summary', '')}\n"
            context += f"Explanation: {content.get('explanation', '')}\n"
            
            # Include more full text if available
            if 'full_text' in content:
                text_sample = content['full_text'][:1000]  # Increased sample size
                context += f"Content: {text_sample}...\n"
    
    if not pdf_found:
        return "❌ No PDF content found. Please upload PDFs first."
    
    prompt = f"""You are an expert assistant helping with questions about uploaded PDF documents.

Question: {question}

Relevant Context from uploaded PDFs:
{context}

IMPORTANT: Answer ONLY based on the provided PDF context above. 
If the context doesn't contain the answer, say "I couldn't find the answer in the uploaded PDFs."
Do not use any external knowledge.

Answer:"""

    model = genai.GenerativeModel(TEXT_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.1,  # Lower temperature for more accurate answers
            "top_p": 0.8,
            "max_output_tokens": 1024,
        }
    )
    
    return response.text.strip()


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application"""
    
    print("\n" + "=" * 60)
    print("🎓 EDUCATION CREATOR WITH VECTOR DATABASE")
    print("   Smart Search • Duplicate Detection • Knowledge Graph")
    print("=" * 60)
    
    # Initialize Vector Database
    db = SimpleVectorDB()
    
    # Show stats
    stats = db.get_stats()
    print(f"\n📊 Database: {stats['total_entries']} topics")
    if stats['latest']:
        print(f"   Latest: {stats['latest']}")
    
    while True:
        print("\n📚 MENU:")
        print("1. Learn new topic (generate + store)")
        print("2. Upload PDF file")
        print("3. Search similar topics")
        print("4. View all topics")
        print("5. Check for duplicates")
        print("6. Ask question about PDFs")
        print("7. Generate image for topic")
        print("8. Database statistics")
        print("9. Exit")
        
        choice = input("\nChoose: ").strip()
        
        if choice == "9":
            print("\n👋 Goodbye!")
            break
        
        try:
            if choice == "1":
                # LEARN NEW TOPIC
                topic = input("\n📝 Topic: ").strip()
                if not topic:
                    print("❌ Topic required!")
                    continue
                
                # Check for duplicates
                duplicates = db.find_duplicates(topic)
                if duplicates:
                    print(f"\n⚠️  Similar topics found:")
                    for dup in duplicates:
                        print(f"  • {dup['topic']} ({dup['similarity']*100:.1f}% similar)")
                    
                    proceed = input("\nContinue anyway? (y/n): ").strip().lower()
                    if proceed != 'y':
                        continue
                
                # Generate content
                print("\n🔄 Generating content...")
                content = generate_text(topic)
                display_content(content, topic)
                
                # Add to database
                db.add(topic, content)
                
                # Save to JSON file
                filename = f"{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump({
                        "topic": topic,
                        "timestamp": datetime.now().isoformat(),
                        "content": content
                    }, f, indent=2)
                print(f"💾 Saved: {filename}")
            
            elif choice == "2":
                # UPLOAD PDF
                print("\n📁 PDF UPLOAD")
                print("Enter the full path to your PDF file:")
                pdf_path = input("PDF path: ").strip().strip('"')
                
                if not pdf_path:
                    print("❌ Path required!")
                    continue
                
                if not os.path.exists(pdf_path):
                    print("❌ File not found!")
                    continue
                
                if not pdf_path.lower().endswith('.pdf'):
                    print("❌ Must be a PDF file!")
                    continue
                
                try:
                    print(f"\n📖 Reading PDF: {os.path.basename(pdf_path)}")
                    
                    # Extract text from PDF
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_text = extract_text_from_pdf(pdf_file)
                    
                    if not pdf_text.strip():
                        print("❌ No text found in PDF!")
                        continue
                    
                    print(f"✅ Extracted {len(pdf_text)} characters")
                    
                    # Process the content
                    print("🔄 Processing PDF content...")
                    filename = os.path.basename(pdf_path)
                    content = process_pdf_content(pdf_text, filename)
                    topic = content['title']
                    
                    # Check for duplicates
                    duplicates = db.find_duplicates(topic)
                    if duplicates:
                        print(f"\n⚠️  Similar topics found:")
                        for dup in duplicates:
                            print(f"  • {dup['topic']} ({dup['similarity']*100:.1f}% similar)")
                        
                        proceed = input("\nContinue anyway? (y/n): ").strip().lower()
                        if proceed != 'y':
                            continue
                    
                    # Display content
                    display_content(content, topic)
                    
                    # Add to database
                    db.add(topic, content)
                    
                    # Save to JSON file
                    json_filename = f"{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(json_filename, 'w') as f:
                        json.dump({
                            "topic": topic,
                            "timestamp": datetime.now().isoformat(),
                            "content": content,
                            "source": "PDF",
                            "source_file": filename
                        }, f, indent=2)
                    print(f"💾 Saved: {json_filename}")
                    
                except Exception as e:
                    print(f"❌ Error processing PDF: {e}")
            
            elif choice == "3":
                # SEARCH
                query = input("\n🔍 Search query: ").strip()
                if not query:
                    continue
                
                results = db.search(query, top_k=5)
                
                if not results:
                    print("❌ No results found")
                    continue
                
                print("\n📋 SEARCH RESULTS:")
                print("=" * 60)
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['topic']}")
                    print(f"   Similarity: {result['similarity']*100:.1f}%")
                    print(f"   Summary: {result['content'].get('summary', 'N/A')[:100]}...")
            
            elif choice == "4":
                # VIEW ALL
                topics = db.get_all_topics()
                if not topics:
                    print("\n📂 Database is empty")
                else:
                    print(f"\n📋 ALL TOPICS ({len(topics)}):")
                    print("=" * 60)
                    for i, topic in enumerate(topics, 1):
                        print(f"  {i}. {topic}")
            
            elif choice == "5":
                # CHECK DUPLICATES
                topic = input("\n📝 Topic to check: ").strip()
                if not topic:
                    continue
                
                duplicates = db.find_duplicates(topic, threshold=0.7)
                
                if not duplicates:
                    print("✅ No similar topics found")
                else:
                    print(f"\n⚠️  Found {len(duplicates)} similar topics:")
                    for dup in duplicates:
                        print(f"  • {dup['topic']} ({dup['similarity']*100:.1f}% similar)")
            
            elif choice == "6":
                # ASK QUESTION ABOUT PDFs
                print("\n❓ QUESTION ABOUT UPLOADED PDFs")
                question = input("Your question: ").strip()
                
                if not question:
                    print("❌ Question required!")
                    continue
                
                print("\n🔍 Searching PDFs...")
                answer = answer_question_from_pdfs(question, db)
                print(f"\n💡 ANSWER:\n{answer}")
            
            elif choice == "7":
                # GENERATE IMAGE
                topic = input("\n📝 Topic: ").strip()
                
                if not topic:
                    print("❌ Topic required!")
                    continue
                
                # Automatically create a descriptive prompt from the topic
                auto_prompt = f"Educational diagram of {topic}, clean minimalist style, white background, professional infographic, clear and simple illustration"
                
                print(f"🎨 Generating image for: {topic}")
                filename = generate_image_huggingface(auto_prompt, topic)
                print(f"\n✅ Image saved: {filename}")
            
            elif choice == "8":
                # STATISTICS
                stats = db.get_stats()
                print("\n📊 DATABASE STATISTICS:")
                print("=" * 60)
                print(f"Total Topics: {stats['total_entries']}")
                print(f"Latest Topic: {stats['latest']}")
                
                if stats['topics']:
                    print(f"\nAll Topics:")
                    for topic in stats['topics']:
                        print(f"  • {topic}")
            
            else:
                print("❌ Invalid choice!")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
