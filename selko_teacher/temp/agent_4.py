import os
import json
import asyncio
import edge_tts

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import FunctionTool

from .util import load_instruction_from_file
from .fetch_yle_article import get_article

# -------------------------
# Config
# -------------------------
MODEL = "gemini-2.5-flash-lite"

# -------------------------
# Tool: Edge TTS
# -------------------------
async def edge_tts_speak(
    text: str,
    voice: str = "fi-FI-SelmaNeural",
    out_path: str = "selko_narration.mp3"
):
    """
    ADK tool: turn Finnish narration text into an mp3 using edge-tts.
    Saves mp3 in working dir and returns filename.
    """
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(out_path)
    return {"audio_file": out_path, "voice": voice}

edge_tts_tool = FunctionTool(edge_tts_speak)

# -------------------------
# 1) Retriever agent
# -------------------------
retriever_agent = LlmAgent(
    model=MODEL,
    name="news_retriever",
    instruction=load_instruction_from_file("news_extraction_instruction.txt"),
    tools=[get_article],
    output_key="article",   # stored in state as ctx.state["article"]
)

# -------------------------
# 2) Narrator agent
# -------------------------
narrator_agent = LlmAgent(
    model=MODEL,
    name="selko_narrator",
    instruction=(
        "You are a Finnish news narrator.\n"
        "You receive the full article text in Finnish in the state key 'article'.\n"
        "Rewrite it into a spoken-friendly form:\n"
        "- Short clear sentences.\n"
        "- Keep meaning, simplify slightly.\n"
        "- Output ONLY valid JSON, no markdown:\n"
        "{ \"narration_fi\": \"...\" }\n"
    ),
    tools=[],
    output_key="narration",   # ctx.state["narration"]
)

# -------------------------
# 3) TTS agent (calls edge_tts_tool)
# -------------------------
tts_agent = LlmAgent(
    model=MODEL,
    name="selko_tts",
    instruction=(
        "You are a Text-to-Speech assistant.\n"
        "You receive JSON in state key 'narration' with field narration_fi.\n"
        "Call the tool edge_tts_speak(text=...) to create an mp3.\n"
        "Return ONLY JSON:\n"
        "{ \"audio_file\": \"<filename>\" }\n"
    ),
    tools=[edge_tts_tool],
    output_key="audio",   # ctx.state["audio"]
)

# -------------------------
# 4) Grammar agent
# -------------------------
grammar_agent = LlmAgent(
    model=MODEL,
    name="selko_grammar_teacher",
    instruction=load_instruction_from_file("grammar_instruction.txt"),
    tools=[],
    output_key="grammar_analyzed"
)

# -------------------------
# 5) Quiz agent
# -------------------------
quiz_agent = LlmAgent(
    model=MODEL,
    name="selko_quiz_maker",
    instruction=load_instruction_from_file("quiz_instruction.txt"),
    tools=[],
    output_key="quiz_question"
)

# -------------------------
# Pipeline
# -------------------------
pipeline_agent = LoopAgent(
    name="learning_pipeline",
    sub_agents=[
        retriever_agent,
        narrator_agent,
        tts_agent,       # <-- new
        grammar_agent,
        quiz_agent
    ],
    max_iterations=1,
)

root_agent = pipeline_agent
