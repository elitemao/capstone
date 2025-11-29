from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import google_search, FunctionTool,agent_tool, ToolContext
from .util import load_instruction_from_file
from .extractContentFrYleUrl import get_article

import edge_tts
import re
import os
import os.path
import uuid
# -------------------------
# Config
# -------------------------

MODEL = "gemini-2.5-flash-lite"  # adjust if needed
#MODEL = "gemini-2.5-flash"

location_startAgent="/Volumes/personal1T/code/capstone/"
# -------------------------
# Tool: Edge TTS
# -------------------------
async def edge_tts_generate_and_serve(
    text: str,
    voice: str = "fi-FI-SelmaNeural",
    rate: str = "+0%",
    **kwargs,
):
    """
    Generates MP3 and returns the local file URL for ADK playback.
    
    Args:
        text: The sentence text to convert to speech.
        
        rate: The speech rate adjustment. MUST be a string in the format [+-]N%, 
              where N is a number. 
              Examples: '-20%', '+10%', '0%'. 
              Default is '-50%' for slow speech.
        
        voice: The voice model to use. Default is fi-FI-SelmaNeural.
    """
    
    tool_context = kwargs.get('tool_context')
	
    # 1. GENERATE FILE NAME
    random_id = uuid.uuid4().hex[:6]
    file_name = f"selko_narration_{random_id}.mp3"
    
    # 🛑 SAVE LOCATION: Ensure we save to the 'static' folder
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(static_dir, exist_ok=True)
    out_path = os.path.join(static_dir, file_name)
        
    # 2. Generate and save the file
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(out_path)
    
    close_fn = getattr(communicate, "close", None)
    if callable(close_fn):
        await close_fn()
    
    # 🛑 URL FIX: Use the /static/ web path
    # This is the standard web path for files saved in the 'static' folder.
    local_url = f"/static/{file_name}"
   	
    return {
        "ok": True, 
        "audio_asset": {
            "url": local_url,
            "display_name": file_name
        }
    }

edge_tts_full_tool = FunctionTool(edge_tts_generate_and_serve)

# Replace the old tts_tool definition with the new FunctionTool
tts_direct_tool = edge_tts_full_tool # Rename for clarity

# -------------------------
# Internal agents: analyst, grammar, quiz
# -------------------------

# 1) Analyst: uses the RSS tool and outputs JSON only
retriever_agent = LlmAgent(
    model=MODEL,
    name="news_retriever",
    instruction=(
        load_instruction_from_file("news_singleContent_instruction.txt")
        + "\n\nIMPORTANT: You are a background worker. "
          "Do NOT talk to the user. "
          "Return ONLY the article content needed by the router. "
          "No greetings, no questions, no suggestions."
    ),
    tools=[get_article],
    output_key="article",
)

freeText_agent = LlmAgent(
    model=MODEL,
    name="free_text_noHeading",
    instruction=(
        load_instruction_from_file("freeText_noHeading_instruction.txt")
        + "\n\nIMPORTANT: You are a background worker. "
          "Do NOT talk to the user. "
          "Return ONLY the article content needed by the router. "
          "No greetings, no questions, no suggestions."
    ),
    tools=[],
    output_key="article",
)



# 2) Grammar agent: receives that JSON and explains grammar in Markdown
grammar_agent = LlmAgent(
    model=MODEL,
    name="selko_grammar_teacher",
    instruction=load_instruction_from_file("grammar_instruction_2.txt")
    + "\n\nIMPORTANT: You are a background worker. "
          "Do NOT talk to the user. "
          "Return ONLY grammar analysis text.",
    tools=[],
    output_key="grammar_analyzed"
)

# 3) Quiz agent: receives the same JSON and creates a quiz in Markdown
quiz_agent = LlmAgent(
    model=MODEL,
    name="selko_quiz_maker",
    instruction=load_instruction_from_file("quiz_instruction_3.txt") 
    + "\n\nIMPORTANT: You are a background worker. "
          "Do NOT talk to the user. "
          "Return ONLY the quiz.",
    tools=[],
    output_key="quiz_question"
)

sentence_agent=LlmAgent(
	model=MODEL,
	name="split_sentence",
	instruction=load_instruction_from_file("split_instruction_1.txt")
	+ "\n\nIMPORTANT: You are a background worker. "
          "Do NOT talk to the user. "
          "Return ONLY a numbered list of sentences.",
	tools=[],
	output_key="sentence_list"
)

retriever_tool = agent_tool.AgentTool(agent=retriever_agent)
freeText_tool = agent_tool.AgentTool(agent=freeText_agent)
grammar_tool   = agent_tool.AgentTool(agent=grammar_agent)
quiz_tool      = agent_tool.AgentTool(agent=quiz_agent)

split_tool     = agent_tool.AgentTool(agent=sentence_agent)


# -------------------------
#  TTS agent 
# -------------------------
# --- Define a new function tool that acts as a bridge ---
async def extract_audio_file(
    input_result: dict,
    **kwargs,
):
    """Extracts the audio_file path from the edge_tts_make_mp3 output."""
    # This tool simply passes the 'audio_file' value to the next tool.
    return {"local_path": input_result.get("audio_file")}

extract_tool = FunctionTool(extract_audio_file)

# -------------------------
# TTS agent (Simple LlmAgent)
# -------------------------
tts_agent = LlmAgent(
    model=MODEL,
    name="selko_tts",
    instruction=(
        load_instruction_from_file("tts_instruction_SIMPLE.txt") # Create a new, simple instruction
    ),
    # Only one tool that does everything!
    tools=[edge_tts_full_tool],
    output_key="audio",
)

tts_tool = agent_tool.AgentTool(agent=tts_agent)


#pipeline_agent = LoopAgent(
#    name="learning_pipeline",
#    sub_agents=[retriever_agent, grammar_agent, quiz_agent],
#    max_iterations=1,
#)

#pipeline_tool = AgentTool(pipeline_agent)

router_agent = LlmAgent(
    model=MODEL,
    name="router",
    instruction=load_instruction_from_file("router_instruction_10.txt"),
    tools=[retriever_tool, freeText_tool, grammar_tool, quiz_tool, tts_direct_tool, split_tool],
    output_key="router_reply"   # enables state writes
)

root_agent = router_agent


