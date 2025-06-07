# Complete agent.py Code

```python
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
    # This tells the agent to join ALL rooms
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        auto_subscribe=AutoSubscribe.AUDIO_ONLY,
        room_prefix=""  # Accept all rooms
    ))
```

## Steps to Update:

1. **Go to your GitHub repository**
2. **Click on `agent.py`**
3. **Click the pencil icon to edit**
4. **Replace the entire content** with the code above
5. **Commit the changes**

## What Changed:

1. **Import AutoSubscribe**: Added proper import for audio subscription
2. **Updated entrypoint**: Added `auto_subscribe=AutoSubscribe.AUDIO_ONLY` to the connect method
3. **Updated main section**: Added `auto_subscribe` and `room_prefix=""` to accept all rooms
4. **Better logging**: Added more detailed logs to track connection status

## After Updating:

1. Railway will automatically redeploy (takes ~1-2 minutes)
2. Watch the Railway logs for:
   ```
   Agent connecting to room: [room-name]
   Agent connected successfully!
   Agent session started, ready for conversation!
   ```
3. Reconnect to LiveKit Meet
4. Your agent should now properly handle audio!

This configuration ensures your agent:
- ✅ Joins any room automatically
- ✅ Only subscribes to audio (not video)
- ✅ Properly handles the audio stream
- ✅ Logs everything for debugging
