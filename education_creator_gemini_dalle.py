"""
Multimodal Education Creator (FREE APIs)
1. Text only (Gemini Flash)
2. Image only (Stable Diffusion – Hugging Face)
3. Text + Image
"""

import os
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from google import genai

# ---------------- LOAD ENV ----------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY missing")

if not HF_API_KEY:
    raise RuntimeError("❌ HUGGINGFACE_API_KEY missing")

# ---------------- GEMINI CONFIG (NEW SDK, FREE SAFE) ----------------
client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-1.5-flash"  # ✅ free-tier supported

# ---------------- HUGGING FACE CONFIG ----------------
HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "stabilityai/stable-diffusion-2-1"
)


HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "image/png"
}

# ---------------- TEXT GENERATION ----------------
def generate_text(topic: str) -> dict:
    prompt = f"""
You are an expert educational content creator.

Topic: {topic}

Return ONLY valid JSON in this exact format:
{{
  "explanation": "5-7 simple lines",
  "realLifeExample": "clear real-life example",
  "stepByStep": ["step 1", "step 2", "step 3"],
  "keyPoints": ["point 1", "point 2", "point 3"],
  "summary": "3 line summary",
  "flashcard": {{
    "question": "question",
    "answer": "answer"
  }},
  "imagePrompt": "simple educational diagram prompt"
}}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    text = response.text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError("❌ Gemini did not return valid JSON")

    return json.loads(text[start:end])

# ---------------- IMAGE GENERATION (RETRY SAFE) ----------------
def generate_image(prompt: str, topic: str) -> str:
    payload = {"inputs": prompt}

    for attempt in range(3):
        response = requests.post(
            HF_MODEL_URL,
            headers=HF_HEADERS,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            filename = f"{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename

        if response.status_code in (503, 504):
            print("⏳ Model loading, retrying in 20 seconds...")
            time.sleep(20)
            continue

        raise RuntimeError(
            f"HF Error {response.status_code}: {response.text}"
        )

    raise RuntimeError("❌ Stable Diffusion failed after retries")

# ---------------- DISPLAY ----------------
def display_content(content: dict, topic: str):
    print("\n" + "=" * 60)
    print(f"📚 TOPIC: {topic.upper()}")
    print("=" * 60)

    print("\n📖 Explanation:")
    print(content["explanation"])

    print("\n🌍 Real Life Example:")
    print(content["realLifeExample"])

    print("\n🪜 Steps:")
    for i, step in enumerate(content["stepByStep"], 1):
        print(f"{i}. {step}")

    print("\n💡 Key Points:")
    for p in content["keyPoints"]:
        print(f"• {p}")

    print("\n📋 Summary:")
    print(content["summary"])

    print("\n🎴 Flashcard:")
    print("Q:", content["flashcard"]["question"])
    print("A:", content["flashcard"]["answer"])

# ---------------- MAIN CLI ----------------
def main():
    print("\n🎓 MULTIMODAL EDUCATION CREATOR (FREE)")
    print("Gemini Flash + Stable Diffusion")
    print("=" * 60)

    while True:
        print("\nMenu:")
        print("1. Text generation only")
        print("2. Image generation only")
        print("3. Text + Image")
        print("4. Exit")

        choice = input("Choose (1-4): ").strip()

        if choice == "4":
            print("👋 Bye Bhai!")
            break

        topic = input("\nEnter topic: ").strip()
        if not topic:
            print("❌ Topic cannot be empty")
            continue

        try:
            if choice == "1":
                content = generate_text(topic)
                display_content(content, topic)

            elif choice == "2":
                prompt = input("Enter image prompt: ").strip()
                img = generate_image(prompt, topic)
                print(f"✅ Image saved: {img}")

            elif choice == "3":
                content = generate_text(topic)
                display_content(content, topic)

                print("\n🎨 Generating image...")
                img = generate_image(content["imagePrompt"], topic)
                print(f"✅ Image saved: {img}")

            else:
                print("❌ Invalid option")

        except Exception as e:
            print("❌ Error:", e)

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
