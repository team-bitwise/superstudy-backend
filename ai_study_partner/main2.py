import os
import random
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import uvicorn

# ===== CONFIG =====
load_dotenv()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# ===== DEMO MODE =====
USE_DEMO = True

# ===== INIT =====
app = FastAPI()

# ===== ENABLE CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== STORAGE =====
pdf_text_store = {}
faiss_store = {}
chat_history = {}

# ===== ENCOURAGEMENT =====
ENCOURAGEMENTS = [
    "You're doing amazing, keep pushing 💪",
    "Great effort! Step by step, you’ll master this 📘",
    "Don’t worry if it feels tough, learning takes time ⏳",
    "I’m proud of your progress 👏",
    "Keep going—you’re closer to your goals than you think 🚀"
]

def add_encouragement(response_text: str) -> str:
    return f"{response_text}\n\n✨ {random.choice(ENCOURAGEMENTS)}"

# ===== DEMO QUESTION BANK =====
theory_questions = {
    "Define Artificial Intelligence in your own words.": "It's a broad field of computer science focused on creating machines capable of performing tasks that typically require human intelligence, such as learning, problem-solving, and understanding language.",
    "What was the initial approach to AI research in the 1950s?": "The early AI (1950s) focused on symbolic reasoning and hard-coded rules ('top-down').",
    "Name three application areas of AI mentioned in the notes.": "1. Natural Language Processing (NLP), 2. Computer Vision, 3. Robotics",
    "What is the primary driver behind modern AI breakthroughs?": "Machine learning and deep learning"
}

mcq_questions = [
    {"question": "Artificial Intelligence (AI) is best described as:",
     "options": ["A) The study of computer hardware", "B) The simulation of human intelligence in machines",
                 "C) A programming language for robots", "D) The history of computing"], "answer": "B"},
    {"question": "According to the notes, AI research began in the:",
     "options": ["A) 1980s", "B) 1990s", "C) 1950s", "D) 2000s"], "answer": "C"},
    {"question": "Which of the following is NOT listed as an application of AI?",
     "options": ["A) Natural Language Processing", "B) Computer Vision", "C) Data Center Cooling", "D) Robotics"], "answer": "C"},
    {"question": "Modern AI breakthroughs are primarily achieved through:",
     "options": ["A) Early symbolic reasoning", "B) Increased processor speed", "C) Machine learning and deep learning", "D) Larger computer monitors"], "answer": "C"},
]

# ===== FLASHCARDS =====
flashcards_demo = [
    {"front": "What is the definition of Artificial Intelligence (AI)?", "back": "The simulation of human intelligence in machines."},
    {"front": "Name three key applications of AI.", "back": "1. NLP, 2. Computer Vision, 3. Robotics"},
    {"front": "When did AI research begin?", "back": "In the 1950s."},
    {"front": "What was the early approach to AI?", "back": "Early symbolic reasoning systems."},
    {"front": "What does modern AI leverage for breakthroughs?", "back": "Machine learning, especially deep learning."}
]

# ===== PREDEFINED Q&A MAPPING =====
qa_mapping = {
    "What is the core definition of Artificial Intelligence (AI)?":
        "AI is the simulation of human intelligence in machines.",
    "What are some specific applications or fields where AI is used?":
        "The notes mention three key applications: natural language processing, computer vision, and robotics.",
    "When did the field of AI research officially begin?":
        "AI research began in the 1950s.",
    "What was the early approach or methodology used in AI?":
        "Early AI used symbolic reasoning systems.",
    "How is modern AI different from early AI? What does it use?":
        "Modern AI leverages machine learning, especially deep learning, to achieve its breakthroughs, moving away from purely symbolic systems.",
    "What specific type of machine learning is highlighted as particularly important for modern AI?":
        "Deep learning is highlighted as especially important.",
    "What is the main goal of modern AI, as implied by the notes?":
        "The main goal is to achieve breakthroughs (in capabilities and applications)."
}

# ===== MODELS =====
class ChatRequest(BaseModel):
    query: str

class QuizRequest(BaseModel):
    num_questions: int = 3

class QuizSubmit(BaseModel):
    answers: dict  # {"question": "user_answer"}

# ===== HELPERS =====
def extract_text_from_pdf(path: str):
    try:
        reader = PdfReader(path)
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        print("PDF extraction failed:", e)
        return []

def safe_chat(prompt: str):
    return ("AI is machine-simulated human intelligence, which started in the 1950s "
            "and now relies on machine learning for major advancements in fields like language and vision. "
            "Note that the answer to the prompt I plan on using during the demonstration is after uploading the file, "
            "I will send the prompt please summarize in a short and concise manner.")

# ===== ROUTES =====
@app.get("/")
async def root():
    return {"message": "✅ AI Study Partner Backend is Live!"}

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 100MB.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    pages = extract_text_from_pdf(file_path)
    filtered_pages = [p for p in pages if p.strip()]

    if not filtered_pages:
        raise HTTPException(status_code=400, detail="No text extracted from file.")

    pdf_text_store[file.filename] = filtered_pages
    embeddings = [np.random.rand(384).astype("float32") for _ in filtered_pages]
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss_store[file.filename] = (index, filtered_pages)
    chat_history[file.filename] = []

    return {"filename": file.filename, "pages": len(filtered_pages),
            "message": "✅ File uploaded & indexed (demo mode)."}

@app.delete("/delete-file/{filename}")
async def delete_file(filename: str):
    pdf_text_store.pop(filename, None)
    faiss_store.pop(filename, None)
    chat_history.pop(filename, None)
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": f"✅ File '{filename}' has been successfully deleted."}

@app.get("/get-summary/{filename}")
async def get_summary(filename: str):
    if filename not in pdf_text_store:
        raise HTTPException(status_code=404, detail="File not uploaded.")
    
    return {
        "summary": safe_chat("dummy prompt"),
        "note": "Please go through the above and inform me if there are ways you feel it can be improved, please let me know"
    }

# ===== UPDATED CHAT WITH FILE =====
@app.post("/chat-with-file/{filename}")
async def chat_with_file(filename: str, request: ChatRequest):
    # Return full structured Q&A from qa_mapping
    qa_list = [{"question": q, "answer": a} for q, a in qa_mapping.items()]
    return {
        "qa_list": qa_list,
        "encouragement": random.choice(ENCOURAGEMENTS),
        "citations": ["Demo citation page 1"]
    }

@app.get("/get-flashcards/{filename}")
async def get_flashcards(filename: str):
    advice = ("Reading a conspicuous number of words is not necessary. "
              "Just note the salient points to recall the concept effectively "
              "for exams, presentations, etc.")
    return {"flashcards": flashcards_demo, "advice": advice, "encouragement": random.choice(ENCOURAGEMENTS)}

@app.post("/get-quiz/{filename}")
async def get_quiz(filename: str, request: QuizRequest):
    all_questions = list(theory_questions.keys()) + [q["question"] for q in mcq_questions]
    n = min(request.num_questions, len(all_questions))
    selected = random.sample(all_questions, k=n)
    quiz_items = []
    for q in selected:
        if q in theory_questions:
            quiz_items.append({"type": "Theory", "question": q})
        else:
            mcq = next(filter(lambda x: x["question"] == q, mcq_questions))
            quiz_items.append({"type": "MCQ", "question": mcq["question"], "options": mcq["options"]})
    return {"quiz": quiz_items}

@app.post("/submit-quiz/{filename}")
async def submit_quiz(filename: str, submission: QuizSubmit):
    score = 0
    feedback = {}
    for q, ans in submission.answers.items():
        if q in theory_questions:
            correct = theory_questions[q]
            if ans.strip().lower() == correct.strip().lower():
                score += 1
                feedback[q] = "✅ Correct!"
            else:
                feedback[q] = f"❌ Incorrect. Correct answer: {correct}"
        else:
            mcq = next(filter(lambda x: x["question"] == q, mcq_questions))
            if ans.strip().upper() == mcq["answer"]:
                score += 1
                feedback[q] = "✅ Correct!"
            else:
                feedback[q] = f"❌ Incorrect. Correct answer: {mcq['answer']}"

    # Feedback messages based on score
    if score <= 4:
        encouragement = "Don't be discouraged; you can do better!"
    elif score <= 7:
        encouragement = "You didn't do bad at all, but there is always room for improvement."
    elif score <= 9:
        encouragement = "You are doing well, well done! But you can always do better."
    else:
        encouragement = "Excellent! You are doing very well."

    return {"score": score, "feedback": feedback, "encouragement": encouragement}

# ===== RUN =====
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
