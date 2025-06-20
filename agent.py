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

# Supabase imports
from supabase import create_client, Client

# LiveKit imports - SIMPLIFIED
try:
    from livekit import api
except ImportError:
    api = None
    logging.warning("LiveKit not fully installed - voice features disabled")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for backend
supabase: Client = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
    logger.info("Supabase client initialized")
else:
    logger.warning("Supabase credentials not found - database features disabled")

# Create FastAPI app
app = FastAPI(title="Ghana Teacher Education Voice Agent")

# PERMANENT CORS CONFIGURATION - Updated with your exact URLs
origins = [
    # Local development
    "http://localhost:3000",
    "http://localhost:3001",
    
    # Your specific Vercel deployment URLs
    "https://ghana-teacher-voice-ijdkm98qq-matthew-nyaabas-projects.vercel.app",
    "https://ghana-teacher-voice.vercel.app",
    "https://ghanateachervoice.vercel.app",
    
    # Pattern for all your Vercel preview deployments
    "https://ghana-teacher-voice-*-matthew-nyaabas-projects.vercel.app",
    "https://ghanateachervoice-*-matthew-nyaabas-projects.vercel.app",
    
    # Additional Vercel patterns
    "https://ghana-teacher-voice-*.vercel.app",
    "https://ghanateachervoice-*.vercel.app",
    
    # Allow all origins for testing (remove in production)
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Handle preflight requests
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"status": "ok"}

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

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None  # Added for database saving
    user_role: str = "student"
    user_program: Optional[str] = None
    user_year: Optional[int] = None
    custom_gpt_id: Optional[str] = None
    custom_instructions: Optional[str] = None
    conversation_history: Optional[List[Dict]] = None

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

2. YEAR 2 COURSES:
   Semester 1:
   - PED 211: Principles and Methods of Teaching (3 credits)
   - ASE 211: Assessment in Education (3 credits)
   - INC 211: Inclusive Education (3 credits)
   - SUB 211: Subject Specialization I (3 credits)
   
   Semester 2:
   - CLM 221: Classroom Management (3 credits)
   - EDR 221: Educational Research (3 credits)
   - STS 221: Teaching Practice I (3 credits)
   - SUB 221: Subject Specialization II (3 credits)

3. YEAR 3 COURSES:
   Semester 1:
   - ADV 311: Advanced Pedagogy (3 credits)
   - SCM 311: School and Community (3 credits)
   - SUB 311: Subject Specialization III (3 credits)
   - PRE 311: Preparation for Teaching Practice (2 credits)
   
   Semester 2:
   - STS 321: Extended Teaching Practice (12 credits)
   - Continuous assessment and mentoring

4. YEAR 4 COURSES:
   Semester 1:
   - EDL 411: Educational Leadership (3 credits)
   - ARS 411: Action Research I (3 credits)
   - SUB 411: Advanced Subject Studies (3 credits)
   - ETH 411: Professional Ethics (2 credits)
   
   Semester 2:
   - ARS 421: Action Research II (3 credits)
   - STS 421: Independent Teaching (6 credits)
   - CAP 421: Capstone Project (3 credits)

5. TEACHING PRACTICE STRUCTURE:
   - Year 1: 1 week school observation
   - Year 2: 4 weeks assisted teaching
   - Year 3: 12 weeks teaching practice (off-campus)
   - Year 4: 6 weeks independent teaching

6. SPECIALIZATION PROGRAMS:
   - Early Grade (KG-P3): Focus on play-based learning, phonics, early numeracy
   - Upper Primary (P4-P6): Subject specialization, transition pedagogy
   - JHS (Forms 1-3): Subject expertise, adolescent psychology

7. ASSESSMENT STRUCTURE:
   - Continuous Assessment: 40%
   - End of Semester Exam: 60%
   - Teaching Practice: Pass/Fail with detailed rubrics

8. ASSESSMENT RUBRICS FOR TEACHING PRACTICE:
   - Lesson Planning: 20% (objectives, activities, assessment alignment)
   - Delivery: 30% (communication, classroom management, student engagement)
   - Subject Mastery: 25% (content accuracy, depth of knowledge)
   - Professional Conduct: 25% (punctuality, ethics, collaboration)

9. KEY POLICIES:
   - Inclusive Education Policy: All teachers must accommodate diverse learners
   - Language Policy: Bilingual instruction (local language + English)
   - ICT Policy: Digital literacy integration in all subjects
   - Gender Responsive Pedagogy: Equal opportunities for all learners

10. NATIONAL TEACHERS' STANDARDS (NTS):
    - Professional Values and Attitudes
    - Professional Knowledge
    - Professional Practice
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

# API ENDPOINTS

@app.get("/")
async def root():
    """Root endpoint - shows API is running"""
    return {
        "status": "healthy",
        "service": "ghana-teacher-education-agent",
        "timestamp": datetime.now().isoformat(),
        "message": "Ghana Teacher Education Voice Agent API is running!",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "auth": "/auth/login, /auth/register",
            "chat": "/chat",
            "token": "/token",
            "curriculum": "/api/curriculum/courses, /api/curriculum/standards"
        },
        "version": "1.0.0",
        "database": "connected" if supabase else "not connected"
    }

@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {
        "status": "healthy", 
        "agent": "teacher-education", 
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if supabase else "not connected"
    }

@app.post("/token")
async def create_token(request: TokenRequest) -> Dict[str, str]:
    """Create a LiveKit token for the client"""
    try:
        # Get LiveKit credentials
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        if not api_key or not api_secret or not api:
            logger.error("LiveKit not properly configured")
            # Return mock token for testing
            return {
                "token": f"demo-token-{request.room_name}-{datetime.now().timestamp()}",
                "url": "wss://webrobot-fkeecy3d.livekit.cloud",
                "room": request.room_name
            }
        
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
        # Return demo token as fallback
        return {
            "token": f"demo-token-{request.room_name}-{datetime.now().timestamp()}",
            "url": "wss://webrobot-fkeecy3d.livekit.cloud",
            "room": request.room_name
        }

@app.post("/auth/login")
async def login(request: LoginRequest):
    """Handle user login - check against Supabase"""
    try:
        if supabase:
            # Get user from Supabase
            response = supabase.table('users').select("*").eq('email', request.email).execute()
            
            if response.data and len(response.data) > 0:
                user_data = response.data[0]
                logger.info(f"User logged in: {user_data['email']}")
                return {
                    "token": f"token-{user_data['id']}-{datetime.now().timestamp()}",
                    "user": user_data
                }
            else:
                raise HTTPException(status_code=404, detail="User not found. Please register first.")
        else:
            # Fallback to demo mode if Supabase not connected
            logger.warning("Using demo mode - Supabase not connected")
            user_data = {
                "id": f"demo-user-{random.randint(1000, 9999)}",
                "email": request.email,
                "name": request.email.split("@")[0].title(),
                "role": "teacher" if "teacher" in request.email else "student",
                "institution": "University of Education, Winneba",
                "institution_type": "university",
                "program": "B.Ed Early Grade",
                "year": 2
            }
            
            return {
                "token": f"demo-token-{request.email}-{datetime.now().timestamp()}",
                "user": user_data
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed. Please try again.")

@app.post("/auth/register")
async def register(request: RegisterRequest):
    """Handle user registration - save to Supabase"""
    try:
        if supabase:
            # Check if user already exists
            existing = supabase.table('users').select("*").eq('email', request.email).execute()
            if existing.data and len(existing.data) > 0:
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Create new user
            user_data = {
                "email": request.email,
                "name": request.name,
                "role": request.role,
                "institution": request.institution,
                "institution_type": request.institutionType,
                "program": request.program if request.program else None,
                "year": int(request.year) if request.year else None
            }
            
            response = supabase.table('users').insert(user_data).execute()
            
            if response.data and len(response.data) > 0:
                created_user = response.data[0]
                logger.info(f"New user registered: {created_user['email']}")
                return {
                    "token": f"token-{created_user['id']}-{datetime.now().timestamp()}",
                    "user": created_user
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to create user")
        else:
            # Fallback to demo mode if Supabase not connected
            logger.warning("Using demo mode - Supabase not connected")
            return {
                "token": f"demo-token-{request.email}-{datetime.now().timestamp()}",
                "user": {
                    "id": f"demo-user-{random.randint(1000, 9999)}",
                    "name": request.name,
                    "email": request.email,
                    "role": request.role,
                    "institution": request.institution,
                    "institution_type": request.institutionType,
                    "program": request.program,
                    "year": int(request.year) if request.year else 1
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Handle chat messages and return AI responses with curriculum knowledge"""
    try:
        # Build context from request
        user_context = {
            "role": request.user_role,
            "program": request.user_program,
            "year": request.user_year
        }
        
        # Create conversation prompt
        messages = [
            {
                "role": "system",
                "content": f"""
                You are an AI Teacher Education Assistant for Ghana with deep knowledge of the B.Ed curriculum.
                
                {GHANA_CURRICULUM_CONTEXT}
                
                USER CONTEXT:
                - Role: {request.user_role}
                - Program: {request.user_program or 'General B.Ed'}
                - Year: {request.user_year or 'Not specified'}
                
                CUSTOM INSTRUCTIONS:
                {request.custom_instructions or 'Provide helpful, curriculum-aligned responses.'}
                
                IMPORTANT:
                - Always reference specific course codes (e.g., EPS 111)
                - Use Ghanaian context and examples
                - Be encouraging and supportive
                - Provide practical teaching advice
                - Reference NTS and NTECF when relevant
                - For lesson planning, include objectives, activities, and assessment
                - Use local examples (cedis for math, local foods for fractions)
                """
            }
        ]
        
        # Add conversation history if provided
        if request.conversation_history:
            for msg in request.conversation_history[-5:]:  # Last 5 messages for context
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Get response from OpenAI - UPDATED FOR NEW VERSION
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fallback to a curriculum-based response
            ai_response = f"I understand you're asking about: {request.message}. Let me help you based on Ghana's B.Ed curriculum..."
        
        # Check if web search was mentioned (for metadata)
        used_web_search = "search" in request.message.lower() or "latest" in request.message.lower()
        
        # Save chat messages to database if user_id is provided
        if supabase and request.user_id:
            try:
                # Save user message
                supabase.table('chat_messages').insert({
                    "user_id": request.user_id,
                    "custom_gpt_id": request.custom_gpt_id,
                    "role": "user",
                    "content": request.message,
                    "metadata": {}
                }).execute()
                
                # Save AI response
                supabase.table('chat_messages').insert({
                    "user_id": request.user_id,
                    "custom_gpt_id": request.custom_gpt_id,
                    "role": "assistant",
                    "content": ai_response,
                    "metadata": {
                        "model": "gpt-4",
                        "used_web_search": used_web_search,
                        "curriculum_context": True
                    }
                }).execute()
                
                logger.info(f"Chat messages saved for user {request.user_id}")
            except Exception as e:
                logger.error(f"Failed to save chat messages: {e}")
                # Continue without saving - don't fail the request
        
        return {
            "response": ai_response,
            "model": "gpt-4",
            "used_web_search": used_web_search,
            "curriculum_context": True
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        
        # Fallback response with curriculum knowledge
        fallback_response = f"""
        I understand you're asking about: {request.message}
        
        Based on Ghana's B.Ed curriculum, here's what I can tell you:
        
        The B.Ed program consists of:
        - 4-year structure with 8 semesters
        - Core courses like Educational Psychology (EPS 111), Child Development (EPS 121)
        - Teaching Practice progressing from observation to independent teaching
        - Specializations in Early Grade, Upper Primary, or JHS
        
        For Year 1 students:
        Semester 1: EPS 111, PFC 111, LIT 111, NUM 111
        Semester 2: EPS 121, CUR 121, ICT 121, STS 121
        
        Please try rephrasing your question or ask about specific courses or topics.
        """
        
        return {
            "response": fallback_response,
            "model": "fallback",
            "error": str(e)
        }

@app.get("/api/curriculum/courses")
async def get_curriculum_courses():
    """Get B.Ed curriculum courses"""
    return {
        "year1": {
            "semester1": [
                {"code": "EPS 111", "name": "Educational Psychology", "credits": 3},
                {"code": "PFC 111", "name": "Professional Practice", "credits": 3},
                {"code": "LIT 111", "name": "Literacy Studies I", "credits": 3},
                {"code": "NUM 111", "name": "Numeracy and Problem Solving", "credits": 3}
            ],
            "semester2": [
                {"code": "EPS 121", "name": "Child Development", "credits": 3},
                {"code": "CUR 121", "name": "Curriculum Studies", "credits": 3},
                {"code": "ICT 121", "name": "Educational Technology", "credits": 3},
                {"code": "STS 121", "name": "School Experience I", "credits": 3}
            ]
        },
        "year2": {
            "semester1": [
                {"code": "PED 211", "name": "Principles and Methods of Teaching", "credits": 3},
                {"code": "ASE 211", "name": "Assessment in Education", "credits": 3},
                {"code": "INC 211", "name": "Inclusive Education", "credits": 3},
                {"code": "SUB 211", "name": "Subject Specialization I", "credits": 3}
            ],
            "semester2": [
                {"code": "CLM 221", "name": "Classroom Management", "credits": 3},
                {"code": "EDR 221", "name": "Educational Research", "credits": 3},
                {"code": "STS 221", "name": "Teaching Practice I", "credits": 3},
                {"code": "SUB 221", "name": "Subject Specialization II", "credits": 3}
            ]
        },
        "year3": {
            "semester1": [
                {"code": "ADV 311", "name": "Advanced Pedagogy", "credits": 3},
                {"code": "SCM 311", "name": "School and Community", "credits": 3},
                {"code": "SUB 311", "name": "Subject Specialization III", "credits": 3},
                {"code": "PRE 311", "name": "Preparation for Teaching Practice", "credits": 2}
            ],
            "semester2": [
                {"code": "STS 321", "name": "Extended Teaching Practice", "credits": 12}
            ]
        },
        "year4": {
            "semester1": [
                {"code": "EDL 411", "name": "Educational Leadership", "credits": 3},
                {"code": "ARS 411", "name": "Action Research I", "credits": 3},
                {"code": "SUB 411", "name": "Advanced Subject Studies", "credits": 3},
                {"code": "ETH 411", "name": "Professional Ethics", "credits": 2}
            ],
            "semester2": [
                {"code": "ARS 421", "name": "Action Research II", "credits": 3},
                {"code": "STS 421", "name": "Independent Teaching", "credits": 6},
                {"code": "CAP 421", "name": "Capstone Project", "credits": 3}
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
                {
                    "name": "Professional Values and Attitudes",
                    "standards": [
                        "Commitment to learners and learning",
                        "Professional conduct and ethics",
                        "Collaborative practices"
                    ]
                },
                {
                    "name": "Professional Knowledge",
                    "standards": [
                        "Knowledge of educational frameworks",
                        "Knowledge of learners",
                        "Subject and curriculum knowledge"
                    ]
                },
                {
                    "name": "Professional Practice",
                    "standards": [
                        "Planning and preparation",
                        "Managing the learning environment",
                        "Teaching and learning",
                        "Assessment"
                    ]
                }
            ]
        },
        "ntecf": {
            "name": "National Teacher Education Curriculum Framework",
            "pillars": [
                {
                    "name": "Subject and Curriculum Knowledge",
                    "description": "Deep understanding of subject matter and curriculum"
                },
                {
                    "name": "Pedagogical Knowledge",
                    "description": "Knowledge of how to teach effectively"
                },
                {
                    "name": "Literacy Studies",
                    "description": "English, Ghanaian languages, and digital literacy"
                },
                {
                    "name": "Supported Teaching in Schools",
                    "description": "Practical classroom experience with mentoring"
                }
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
    port = int(os.getenv("PORT", 8081))
    
    # Just run the API server
    logger.info(f"Starting Ghana Teacher Education API on port {port}")
    logger.info(f"Database status: {'Connected' if supabase else 'Not connected - using demo mode'}")
    logger.info("Note: Voice agent features are disabled - running as API server only")
    uvicorn.run(app, host="0.0.0.0", port=port)
