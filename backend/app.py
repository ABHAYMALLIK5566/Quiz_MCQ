from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, APIRouter, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from typing import Optional
import google.generativeai as genai
import fitz
import textract
import re
import random
from pathlib import Path
import tempfile
import os
import chardet

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Database Models (Using SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    api_key = Column(String)
    is_active = Column(Boolean, default=True)

Base.metadata.create_all(bind=engine)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class UserCreate(BaseModel):
    email: str
    password: str
    api_key: str

class Token(BaseModel):
    access_token: str
    token_type: str

class APIKeyUpdate(BaseModel):
    api_key: str

# Utility Functions
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

def authenticate_user(db, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

# Routers
router = APIRouter()

@router.put("/users/me/api-key")
async def update_api_key(
    api_key_update: APIKeyUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    current_user.api_key = api_key_update.api_key
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return {"message": "API key updated successfully"}

# API Endpoints
app = FastAPI()
app.include_router(router)

@app.post("/register", response_model=Token)
async def register(user: UserCreate):
    db = SessionLocal()
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password, api_key=user.api_key)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def extract_text(file_path: Path, page_range: tuple = None) -> str:
    """
    Robustly extracts text from files:
    - Tries PyMuPDF (fitz) for PDF extraction first, regardless of extension.
    - Falls back to textract + chardet encoding detection for other file types or on failure.
    """
    try:
        # Try PDF extraction first, regardless of extension
        try:
            with fitz.open(file_path) as doc:
                print(f"PDF has {len(doc)} pages.")
                if page_range:
                    start_page = max(0, page_range[0] - 1)
                    end_page = min(len(doc), page_range[1])
                    text = " ".join(doc[page].get_text() for page in range(start_page, end_page))
                else:
                    text = " ".join(page.get_text() for page in doc)
                print(f"Extracted PDF text length: {len(text)}")
                if text and text.strip():
                    return text
        except Exception as pdf_error:
            print(f"PDF extraction failed ({pdf_error}), falling back to textract...")

        # Fallback: use textract for non-PDFs or if PDF extraction fails
        raw_data = textract.process(str(file_path))
        print(f"Raw data length: {len(raw_data)} bytes")
        detected = chardet.detect(raw_data)
        print(f"Detected encoding: {detected['encoding']} with confidence {detected['confidence']}")
        try:
            text = raw_data.decode(detected['encoding'] if detected['encoding'] else 'utf-8', errors='replace')
        except Exception as decode_error:
            print(f"Primary decoding failed ({decode_error}), trying latin-1 fallback")
            text = raw_data.decode('latin-1', errors='replace')
        print(f"Extracted text snippet: {text[:200]}...")
        if not text or not text.strip():
            raise ValueError("No text could be extracted. Make sure you are uploading a valid, non-empty file.")
        return text

    except Exception as e:
        print(f"Text extraction failed: {str(e)}")
        raise ValueError(f"Text extraction failed: {str(e)}")

def generate_mcqs(text: str, num_questions: int, difficulty: str) -> str:
    """Generates MCQs using Google Generative AI with chunking and difficulty levels."""
    try:
        chunks, current_chunk, current_length = [], [], 0
        sentence_endings = r'(?<!\b\w\.\w.)(?<!\b[A-Z][a-z]\.)(?<=\.|\?|!)\s+'
        
        # Split text into chunks of ~30,000 characters
        for sentence in re.split(sentence_endings, text):
            sentence_length = len(sentence)
            if current_length + sentence_length > 30000:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0
            current_chunk.append(sentence)
            current_length += sentence_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        all_mcqs = []
        diff_map = {
            'easy': ('e', "Ask basic, clear questions about spiritual concepts from the text. "
                          "Keep language simple and options straightforward."),
            'medium': ('m', "Ask questions requiring interpretation of spiritual ideas. "
                            "Options should be plausible with clear language."),
            'hard': ('h', "Ask deep, thought-provoking questions using synonyms and rephrased concepts. "
                          "Explain special terms and use nuanced options.")
        }
        
        diff_code, diff_instructions = diff_map.get(difficulty.lower(), ('m', ""))

        for chunk in chunks:
            prompt = f"""Generate {num_questions} spiritual MCQs from this text.
For each MCQ, use this format exactly:
Q1. [Question]
a) Option 1
b) Option 2
c) Option 3
d) Option 4
Answer: [Correct letter]

Instructions:
- Focus on spiritual concepts and ideas from the text.
- {diff_instructions}
- Make sure to use this format for every question.

Text: {chunk}"""

            try:
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                # Debugging: Print raw response details
                print("Gemini raw output:", repr(response.text))
                # Check for empty response
                if not response.text:
                    raise ValueError("Gemini returned empty response")
                formatted = format_mcqs(response.text)
                print("Formatted MCQs:", repr(formatted))
                all_mcqs.append(formatted)
            except Exception as e:
                print(f"Error generating MCQs: {str(e)}")
                raise ValueError(f"AI generation failed: {str(e)}")
        return "\n".join(all_mcqs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_mcqs_endpoint(
    file: UploadFile,
    difficulty: str = Form("medium"),
    num_questions: int = Form(10),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user)
):
    # File processing
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    try:
        page_range = (start_page, end_page) if start_page and end_page else None
        text = extract_text(tmp_path, page_range)
        # Validate extracted text
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        # Configure Gemini
        genai.configure(api_key=current_user.api_key)
        # Generate MCQs
        return generate_mcqs(text, num_questions, difficulty)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

def format_mcqs(raw_text: str) -> str:
    """
    Formats MCQs from Gemini output into a clean, readable format.
    Handles outputs with literal '\\n' and ensures proper separation.
    """
    # Convert all literal '\n' (not real newlines) to real newlines
    text = raw_text.replace('\\n', '\n')

    # Pattern to match each MCQ block (question, options, answer)
    pattern = re.compile(
        r'Q\d+\.\s*(.*?)\n'                    # Question (non-greedy)
        r'a\)\s*(.*?)\n'                       # Option a
        r'b\)\s*(.*?)\n'                       # Option b
        r'c\)\s*(.*?)\n'                       # Option c
        r'd\)\s*(.*?)\n'                       # Option d
        r'Answer:\s*([A-Da-d])',               # Answer letter
        re.DOTALL
    )

    matches = pattern.findall(text)
    formatted_mcqs = []
    for idx, (question, a, b, c, d, answer) in enumerate(matches, 1):
        formatted = [
            f"Q{idx}. {question.strip()}",
            f"a) {a.strip()}",
            f"b) {b.strip()}",
            f"c) {c.strip()}",
            f"d) {d.strip()}",
            f"Answer: {answer.upper()}",
            "-" * 50
        ]
        formatted_mcqs.append("\n".join(formatted))

    # If nothing matched, return the cleaned raw text for debugging
    if not formatted_mcqs:
        # Remove excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # Join MCQs with a blank line between each
    return "\n\n".join(formatted_mcqs)