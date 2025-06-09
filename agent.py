import os
import logging
import random
import asyncio
import json
import base64
from typing import Dict, List, Optional
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# FastAPI imports
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# LiveKit imports
from livekit import api, agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, AutoSubscribe
from livekit.plugins import openai, elevenlabs, silero

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Ghana Teacher Education Voice Agent")

# Enable CORS for your Next.js frontend - UPDATED WITH EXACT DOMAINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://ghana-teacher-voice.vercel.app",
        "https://ghanateachervoice.vercel.app",
        "https://ghana-teacher-voice-*.vercel.app",
        "https://ghanateachervoice-*.vercel.app",
        "https://ghana-teacher-voice-matthew-nyaabas-projects.vercel.app",
        "https://ghanateachervoice-ino5l6wrc-matthew-nyaabas-projects.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Request/Response models
class TokenRequest(BaseModel):
    room_name: str
    participant_name: str
    role: str = "student"
    institution: str = ""
    custom_gpt_id: Optional[str] = None
    teacher_profile: Optional[Dict] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    role: str
    institution: str
    institutionType: str
    program: str = None
    year: str = "1"
    semester: str = "1"

# Ghana Curriculum Context
GHANA_CURRICULUM_CONTEXT = """
GHANA B.Ed CURRICULUM KNOWLEDGE BASE:

1. YEAR 1 COURSES:
   Semester 1:
   - EPS 111: Educational Psychology (3 credits)
   - PFC 111: Professional Practice (3 credits)
   - LIT 111: Literacy Studies I (3 credits)
   - NUM 111: Numeracy and Problem Solving (3 credits)
   
   Semester 2:
   - EPS 121: Child Development (3 credits)
   - CUR 121: Curriculum Studies (3 credits)
   - ICT 121: Educational Technology (3 credits)
   - STS 121: School Experience I (3 credits)

2. TEACHING PRACTICE STRUCTURE:
   - Year 1: 1 week school observation
   - Year 2: 4 weeks assisted teaching
   - Year 3: 12 weeks teaching practice (off-campus)
   - Year 4: 6 weeks independent teaching

3. SPECIALIZATION PROGRAMS:
   - Early Grade (KG-P3): Focus on play-based learning, phonics, early numeracy
   - Upper Primary (P4-P6): Subject specialization, transition pedagogy
   - JHS (Forms 1-3): Subject expertise, adolescent psychology

4. ASSESSMENT RUBRICS:
   - Lesson Planning: 20% (objectives, activities, assessment alignment)
   - Delivery: 30% (communication, classroom management, student engagement)
   - Subject Mastery: 25% (content accuracy, depth of knowledge)
   - Professional Conduct: 25% (punctuality, ethics, collaboration)

5. KEY POLICIES:
   - Inclusive Education Policy: All teachers must accommodate diverse learners
   - Language Policy: Bilingual instruction (local language + English)
   - ICT Policy: Digital literacy integration in all subjects
"""

# Document processing capability
class DocumentProcessor:
    """Process and extract content from supporting documents"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        # In production, use PyPDF2 or similar
        # For now, return placeholder
        return "Document content would be extracted here"
    
    @staticmethod
    def process_documents(documents: List[Dict]) -> str:
        """Process all documents and create context"""
        context = "SUPPORTING DOCUMENTS CONTEXT:\n\n"
        
        for doc in documents:
            context += f"Document: {doc['name']}\n"
            # In production, fetch and process actual document
            context += f"Content: [Document content would be processed]\n\n"
        
        return context

# Enhanced Teacher Education Assistant
class TeacherEducationAssistant(Agent):
    def __init__(
        self, 
        custom_instructions: str = None, 
        user_context: dict = None,
        teacher_profile: dict = None,
        supporting_documents: List[Dict] = None
    ) -> None:
        llm = openai.LLM(model="gpt-4o")
        stt = openai.STT()
        
        # Voice selection based on teacher profile
        if teacher_profile and teacher_profile.get('voiceId'):
            # Use teacher's custom voice if available
            voice_id = teacher_profile['voiceId']
            if voice_id.startswith('elevenlabs_'):
                # Use ElevenLabs for custom voices
                eleven_api_key = os.getenv('ELEVEN_API_KEY')
                tts = elevenlabs.TTS(
                    api_key=eleven_api_key,
                    voice_id=voice_id.replace('elevenlabs_', '')
                )
            else:
                # Use OpenAI TTS with voice selection
                tts = openai.TTS(voice=voice_id)
        else:
            # Default TTS
            tts = openai.TTS()
        
        silero_vad = silero.VAD.load()
        
        # Process supporting documents if provided
        document_context = ""
        if supporting_documents:
            document_context = DocumentProcessor.process_documents(supporting_documents)
        
        # Build comprehensive instructions
        base_instructions = f"""
        You are an AI Teacher Education Assistant for Ghana's educational system.
        
        {"TEACHER IDENTITY:" if teacher_profile else ""}
        {f"You are speaking as {teacher_profile['name']}, {teacher_profile['title']} at {teacher_profile['institution']}" if teacher_profile else ""}
        {f"Maintain the personality and teaching style of {teacher_profile['name']}" if teacher_profile else ""}
        
        USER CONTEXT:
        - Role: {user_context.get('role', 'student') if user_context else 'student'}
        - Institution: {user_context.get('institution', 'Not specified') if user_context else 'Not specified'}
        - Program: {user_context.get('program', 'General B.Ed') if user_context else 'General B.Ed'}
        - Year: {user_context.get('year', 'Not specified') if user_context else 'Not specified'}
        
        CUSTOM INSTRUCTIONS:
        {custom_instructions or 'Be helpful and supportive in teaching.'}
        
        {document_context}
        
        {GHANA_CURRICULUM_CONTEXT}
        
        TEACHING APPROACH:
        - Use Ghanaian examples (local foods, cedis, familiar contexts)
        - Reference specific course codes when discussing topics
        - For students: Break down complex topics, provide study tips
        - For teachers: Share implementation strategies
        - Be encouraging and supportive
        - Check for understanding frequently
        """
        
        super().__init__(
            instructions=base_instructions,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero_vad,
        )

# API ENDPOINTS

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ghana-teacher-education-agent",
        "timestamp": datetime.now().isoformat(),
        "message": "Teacher Education Voice Agent API is running"
    }

@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "healthy", "agent": "teacher-education"}

@app.options("/token")
async def token_options():
    """Handle OPTIONS request for CORS preflight"""
    return {"status": "ok"}

@app.post("/token")
async def create_token(request: TokenRequest) -> Dict[str, str]:
    """Create a LiveKit token for the client"""
    try:
        # Get LiveKit credentials
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        if not api_key or not api_secret:
            logger.error("LiveKit credentials missing")
            raise HTTPException(status_code=500, detail="LiveKit credentials not configured")
        
        # Create access token
        token = api.AccessToken(api_key, api_secret)
        
        # Set participant identity and name
        token.with_identity(request.participant_name)
        token.with_name(request.participant_name)
        
        # Add metadata for agent context
        metadata = {
            "role": request.role,
            "institution": request.institution,
            "custom_gpt_id": request.custom_gpt_id,
            "teacher_profile": json.dumps(request.teacher_profile) if request.teacher_profile else None
        }
        
        token.with_metadata(json.dumps(metadata))
        
        # Grant permissions
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=request.room_name,
            can_publish=True,
            can_publish_data=True,
            can_subscribe=True
        ))
        
        # Generate JWT
        jwt_token = token.to_jwt()
        
        logger.info(f"Token created for {request.participant_name} in room {request.room_name}")
        
        return {
            "token": jwt_token,
            "url": os.getenv("LIVEKIT_URL", "wss://webrobot-fkeecy3d.livekit.cloud"),
            "room": request.room_name
        }
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/auth/login")
async def login_options():
    """Handle OPTIONS request for CORS preflight"""
    return {"status": "ok"}

@app.post("/auth/login")
async def login(request: LoginRequest):
    """Handle user login"""
    user_data = {
        "id": f"user-{random.randint(1000, 9999)}",
        "email": request.email,
        "role": "student",
        "institution": "University of Education, Winneba",
        "institutionType": "university",
        "program": "B.Ed Early Grade",
        "year": 2
    }
    
    if "teacher" in request.email:
        user_data["role"] = "teacher"
        user_data["name"] = "Teacher " + request.email.split("@")[0].title()
    elif "admin" in request.email:
        user_data["role"] = "admin"
        user_data["name"] = "Admin " + request.email.split("@")[0].title()
    else:
        user_data["name"] = "Student " + request.email.split("@")[0].title()
    
    return {
        "token": f"demo-token-{request.email}-{datetime.now().timestamp()}",
        "user": user_data
    }

@app.options("/auth/register")
async def register_options():
    """Handle OPTIONS request for CORS preflight"""
    return {"status": "ok"}

@app.post("/auth/register")
async def register(request: RegisterRequest):
    """Handle user registration"""
    return {
        "token": f"demo-token-{request.email}-{datetime.now().timestamp()}",
        "user": {
            "id": f"user-{random.randint(1000, 9999)}",
            "name": request.name,
            "email": request.email,
            "role": request.role,
            "institution": request.institution,
            "institutionType": request.institutionType,
            "program": request.program,
            "year": int(request.year) if request.year else 1
        }
    }

@app.get("/api/curriculum/courses")
async def get_curriculum_courses():
    """Get B.Ed curriculum courses"""
    return {
        "year1": {
            "semester1": [
                "Educational Psychology",
                "Introduction to Teaching",
                "Communication Skills",
                "African Studies"
            ],
            "semester2": [
                "Child Development",
                "Curriculum Studies",
                "Educational Technology",
                "Ghanaian Language"
            ]
        },
        "year2": {
            "semester1": [
                "Teaching Methods",
                "Assessment in Education",
                "Inclusive Education",
                "Subject Specialization I"
            ],
            "semester2": [
                "Classroom Management",
                "Educational Research",
                "Teaching Practice I",
                "Subject Specialization II"
            ]
        }
    }

@app.get("/api/curriculum/standards")
async def get_teaching_standards():
    """Get Ghana teaching standards"""
    return {
        "nts": {
            "name": "National Teachers' Standards",
            "domains": [
                "Professional Values and Attitudes",
                "Professional Knowledge",
                "Professional Practice"
            ]
        },
        "ntecf": {
            "name": "National Teacher Education Curriculum Framework",
            "pillars": [
                "Subject and Curriculum Knowledge",
                "Pedagogical Knowledge", 
                "Literacy Studies",
                "Supported Teaching in Schools"
            ]
        }
    }

# Add endpoint for document upload
@app.post("/upload/document")
async def upload_document(
    file: UploadFile = File(...),
    gpt_id: str = Form(...)
):
    """Upload supporting document for a custom GPT"""
    try:
        # In production, save to cloud storage (S3, GCS, etc.)
        # For now, save locally
        file_path = f"/tmp/{gpt_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document for context
        if file.filename.endswith('.pdf'):
            # Extract text from PDF
            text_content = DocumentProcessor.extract_text_from_pdf(content)
        else:
            text_content = content.decode('utf-8', errors='ignore')
        
        return {
            "filename": file.filename,
            "size": len(content),
            "path": file_path,
            "preview": text_content[:500] + "..." if len(text_content) > 500 else text_content
        }
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add voice cloning endpoint
@app.post("/voice/clone")
async def clone_voice(
    audio: UploadFile = File(...),
    name: str = Form(...)
):
    """Clone teacher's voice from audio sample"""
    try:
        # In production, integrate with ElevenLabs API
        # For now, return mock response
        audio_content = await audio.read()
        
        # Save audio sample
        audio_path = f"/tmp/voice_{name}_{datetime.now().timestamp()}.webm"
        with open(audio_path, "wb") as f:
            f.write(audio_content)
        
        # In production: Send to ElevenLabs for voice cloning
        # voice_id = elevenlabs_client.clone_voice(audio_content, name)
        
        # Mock response
        voice_id = f"custom_voice_{name.lower().replace(' ', '_')}"
        
        return {
            "voice_id": voice_id,
            "name": name,
            "status": "processing",
            "message": "Voice cloning initiated. This usually takes 2-3 minutes."
        }
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main execution - SIMPLIFIED FOR RAILWAY
if __name__ == "__main__":
    # Get port from Railway
    port = int(os.getenv("PORT", 8000))
    
    # Just run the API server
    logger.info(f"Starting Ghana Teacher Education API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
