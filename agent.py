import os
import logging
import random
import asyncio
from typing import Dict
from datetime import datetime

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

# Main execution - SIMPLIFIED FOR RAILWAY
if __name__ == "__main__":
    # Get port from Railway
    port = int(os.getenv("PORT", 8000))
    
    # Just run the API server
    logger.info(f"Starting Ghana Teacher Education API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
