import os
import logging
from dotenv import load_dotenv

load_dotenv()

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, AutoSubscribe
from livekit.plugins import openai, elevenlabs, silero

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Assistant(Agent):
    def __init__(self) -> None:
        llm = openai.LLM(model="gpt-4o")
        stt = openai.STT()
        
        eleven_api_key = os.getenv('ELEVEN_API_KEY')
        tts = elevenlabs.TTS(api_key=eleven_api_key) if eleven_api_key else openai.TTS()
        
        silero_vad = silero.VAD.load()

        super().__init__(
            instructions="""
                You are a helpful assistant communicating 
                via voice. Be friendly and conversational.
            """,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero_vad,
        )

async def entrypoint(ctx: JobContext):
    logger.info(f"Agent connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    logger.info("Agent connected successfully!")
    
    # Create and start agent session
    assistant = Assistant()
    session = AgentSession()
    
    await session.start(
        room=ctx.room,
        agent=assistant
    )
    
    logger.info("Agent session started, ready for conversation!")

if __name__ == "__main__":
    # Simple version that works with your LiveKit version
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
