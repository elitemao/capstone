from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import google_search, FunctionTool,agent_tool, ToolContext
from .util import load_instruction_from_file
from .fetch_yle_article import get_article

import edge_tts
import re
import os
# -------------------------
# Config
# -------------------------

MODEL = "gemini-2.5-flash-lite"  # adjust if needed
#MODEL = "gemini-2.5-flash"


# -------------------------
# Tool: Edge TTS
# -------------------------
import os
import uuid
import edge_tts

async def edge_tts_make_mp3(
    text: str,
    voice: str = "fi-FI-SelmaNeural",
    rate: str = "-50%",
    out_path: str = "selko_narration.mp3",
):
    """
    Generate a local mp3 file and return its path.
    No artifact upload here.
    """
    if out_path is None:
        out_path = f"selko_{uuid.uuid4().hex}.mp3"

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(out_path)

    close_fn = getattr(communicate, "close", None)
    if callable(close_fn):
        await close_fn()

    return {"audio_file": out_path, "voice": voice}

async def edge_tts_upload_artifact(
    local_path: str,
    **kwargs,  # ADK injects tool_context here
):
    """
    Upload an existing local mp3 to ADK artifact service (GCS).
    Returns artifact object with a playable URL.
    """
    tool_context = kwargs.get("tool_context")

    if not tool_context or not getattr(tool_context, "artifact_service", None):
        return {"ok": False, "error": "No artifact_service available."}

    artifact = await tool_context.artifact_service.save_file(
        local_path=local_path,
        mime_type="audio/mpeg",
        display_name=os.path.basename(local_path),
    )

    return {"ok": True, "audio_artifact": artifact}


# -------------------------
# Internal agents: analyst, grammar, quiz
# -------------------------

# 1) Analyst: uses the RSS tool and outputs JSON only
retriever_agent = LlmAgent(
    model=MODEL,
    name="news_retriever",
    instruction=(
        load_instruction_from_file("news_extraction_instruction_1.txt")
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
    name="free_text_dissect",
    instruction=(
        load_instruction_from_file("freeText_instruction.txt")
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
    instruction=load_instruction_from_file("grammar_instruction.txt")
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
    instruction=load_instruction_from_file("quiz_instruction.txt") 
    + "\n\nIMPORTANT: You are a background worker. "
          "Do NOT talk to the user. "
          "Return ONLY the quiz.",
    tools=[],
    output_key="quiz_question"
)

sentence_agent=LlmAgent(
	model=MODEL,
	name="split_sentence",
	instruction=load_instruction_from_file("split_instruction.txt")
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

edge_tts_make_mp3_tool = FunctionTool(edge_tts_make_mp3)
edge_tts_upload_artifact_tool = FunctionTool(edge_tts_upload_artifact)

# -------------------------
#  TTS agent 
# -------------------------
tts_agent = LlmAgent(
    model=MODEL,
    name="selko_tts",
    instruction=(
        load_instruction_from_file("tts_instruction_6.txt")
        + "\n\nWorkflow:\n"
          "1) Call edge_tts_make_mp3_tool with the sentence and playback rate.\n"
          "2) From its result, extract the audio_file variable.\n"
          "3) Call edge_tts_upload_artifact_tool(local_path=audio_file).\n"
          "4) Return ONLY the upload tool result.\n"
    ),
    tools=[edge_tts_make_mp3_tool, edge_tts_upload_artifact_tool],
    output_key="audio",
)

tts_tool       = agent_tool.AgentTool(agent=tts_agent)

#pipeline_agent = LoopAgent(
#    name="learning_pipeline",
#    sub_agents=[retriever_agent, grammar_agent, quiz_agent],
#    max_iterations=1,
#)

#pipeline_tool = AgentTool(pipeline_agent)

router_agent = LlmAgent(
    model=MODEL,
    name="router",
    instruction=load_instruction_from_file("router_instruction_8.txt"),
    tools=[retriever_tool, freeText_tool, grammar_tool, quiz_tool, tts_tool, split_tool],
    output_key="router_reply"   # enables state writes
)

root_agent = router_agent


