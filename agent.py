import os
import logging
import random
import asyncio
from typing import Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# FastAPI imports
from fastapi import FastAPI, HTTPException
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

# Enable CORS for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ghana-education-voice.vercel.app",
        "https://*.vercel.app",
        "*"  # For testing (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TokenRequest(BaseModel):
    room_name: str
    participant_name: str

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

# Ghana Teacher Education Agent
class TeacherEducationAssistant(Agent):
    def __init__(self) -> None:
        llm = openai.LLM(model="gpt-4o")
        stt = openai.STT()
        
        eleven_api_key = os.getenv('ELEVEN_API_KEY')
        tts = elevenlabs.TTS(api_key=eleven_api_key) if eleven_api_key else openai.TTS()
        
        silero_vad = silero.VAD.load()
        
        super().__init__(
            instructions="""
                You are an AI Teacher Education Assistant for Ghana's educational system.
                
                GREETING PROTOCOL:
                1. Welcome the participant warmly in English
                2. Ask: "Which language would you prefer? I can speak English, Twi, Ga, Ewe, or Dagbani"
                3. Ask about their role: "Are you a student teacher, practicing teacher, or education administrator?"
                4. Ask about their institution and current program level
                
                YOUR EXPERTISE:
                - Ghana's B.Ed Teacher Education Curriculum (4-year program)
                - Early Grade (KG-P3), Upper Primary (P4-P6), and JHS specializations
                - Teaching methods and pedagogical approaches
                - Classroom management strategies
                - Assessment and evaluation techniques
                - Ghana Education Service (GES) policies and standards
                - National Teachers' Standards (NTS) and National Teacher Education Curriculum Framework (NTECF)
                
                CONVERSATION GUIDELINES:
                - Always relate responses to Ghana's educational context
                - Reference specific courses in the B.Ed curriculum when relevant
                - Provide practical examples from Ghanaian classrooms
                - Be culturally sensitive and aware of local educational challenges
                - Support student teachers preparing for teaching practice
                - Help with lesson planning using Ghana's curriculum
                - Discuss inclusive education in the Ghanaian context
                
                LANGUAGE SUPPORT:
                - Switch between languages as requested
                - Explain educational concepts in simple terms
                - Use local examples and contexts
                
                Remember: You're not just an AI, you're a supportive colleague in Ghana's teacher education journey!
            """,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero_vad,
        )

# Agent entry point
async def agent_entrypoint(ctx: JobContext):
    logger.info(f"Teacher Education Agent connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    logger.info("Agent connected successfully!")
    
    # Create and start agent session
    assistant = TeacherEducationAssistant()
    session = AgentSession()
    
    await session.start(
        room=ctx.room,
        agent=assistant
    )
    
    logger.info("Teacher Education session started, ready for conversation!")

# API ENDPOINTS

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ghana-teacher-education-agent",
        "timestamp": datetime.now().isoformat(),
        "message": "Teacher Education Voice Agent is running"
    }

@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "healthy", "agent": "teacher-education"}

@app.post("/token")
async def create_token(request: TokenRequest) -> Dict[str, str]:
    """Create a LiveKit token for the client"""
    try:
        # Get LiveKit credentials
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        if not api_key or not api_secret:
            raise HTTPException(status_code=500, detail="LiveKit credentials not configured")
        
        # Create access token
        token = api.AccessToken(api_key, api_secret)
        
        # Set participant identity and name
        token.with_identity(request.participant_name)
        token.with_name(request.participant_name)
        
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
        
        return {
            "token": jwt_token,
            "url": os.getenv("LIVEKIT_URL"),
            "room": request.room_name
        }
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
async def login(request: LoginRequest):
    """Handle user login - returns user data based on email"""
    # For demo: Create user profile based on email patterns
    # In production, this would check a database
    
    user_data = {
        "id": f"user-{random.randint(1000, 9999)}",
        "email": request.email,
        "role": "student",  # Default role
        "institution": "University of Education, Winneba",  # Default institution
        "institutionType": "university",
        "program": "B.Ed Early Grade",
        "year": 2
    }
    
    # Determine user details from email (demo logic)
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

@app.post("/auth/register")
async def register(request: RegisterRequest):
    """Handle user registration"""
    # In production, save to database
    # For now, return success with user data
    
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

# Function to run the agent worker
def run_agent_worker():
    """Run the LiveKit agent worker"""
    try:
        if __name__ == "__main__":
            # Use the CLI when running as main
            cli.run_app(WorkerOptions(entrypoint_fnc=agent_entrypoint))
    except Exception as e:
        logger.error(f"Agent worker error: {e}")

# Main execution
if __name__ == "__main__":
    # Check if we're running as agent or API server
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        # Run as LiveKit agent (original behavior)
        cli.run_app(WorkerOptions(entrypoint_fnc=agent_entrypoint))
    else:
        # Run as API server
        port = int(os.getenv("PORT", 8000))
        
        # Start agent worker in background thread
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(run_agent_worker)
        
        # Start FastAPI server
        logger.info(f"Starting Ghana Teacher Education API on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
