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

async def set_article_text(
    text: str,
    title: str = "",
    ctx: ToolContext | None = None,   # <-- ADK injects ctx
):
    if ctx is None:
        return {"ok": False, "error": "No ctx, cannot write state."}

    if not title.strip():
        first_line = text.strip().splitlines()[0][:80]
        title = first_line if first_line else "User provided text"

    ctx.state["article"] = {
        "title": title,
        "text": text.strip(),
        "source": "user_text",
        "link": None,
    }
    return {"ok": True, "stored_as": "state['article']"}

    
set_article_tool = FunctionTool(set_article_text)

# -------------------------
# Tool: Edge TTS
# -------------------------
async def edge_tts_speak(
    text: str,
    voice: str = "fi-FI-SelmaNeural",
    rate: str = "-50%",                 # keep as str, edge-tts wants "-50%"
    out_path: str = "selko_narration.mp3",
    ctx: ToolContext | None = None,     # <-- ADK injects THIS
):
    """
    Turn narration text into mp3, upload to artifact service if available.
    """
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(out_path)

    close_fn = getattr(communicate, "close", None)
    if callable(close_fn):
        await close_fn()

    if ctx and getattr(ctx, "artifact_service", None):
        artifact = await ctx.artifact_service.save_file(
            local_path=out_path,
            mime_type="audio/mpeg",
            display_name=os.path.basename(out_path),
        )
        return {"audio_artifact": artifact, "voice": voice}

    return {"audio_file": out_path, "voice": voice}

edge_tts_tool = FunctionTool(edge_tts_speak)


# -------------------------
# Internal agents: analyst, grammar, quiz
# -------------------------

# 1) Analyst: uses the RSS tool and outputs JSON only
retriever_agent = LlmAgent(
    model=MODEL,
    name="news_retriever",
    instruction=(
        load_instruction_from_file("news_extraction_instruction.txt")
        + "\n\nIMPORTANT: You are a background worker. "
          "Do NOT talk to the user. "
          "Return ONLY the article content needed by the router. "
          "No greetings, no questions, no suggestions."
    ),
    tools=[get_article],
    output_key="article",
)


# -------------------------
# 1.5) TTS agent (calls edge_tts_tool)
# -------------------------
tts_agent = LlmAgent(
    model=MODEL,
    name="selko_tts",
    instruction=load_instruction_from_file("tts_instruction.txt")
    + "\n\nIMPORTANT: You are a background worker. "
          "Do NOT talk to the user. "
          "Call edge_tts_tool and return ONLY the tool result.",
    tools=[edge_tts_tool],
    output_key="audio",   # ctx.state["audio"]
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
grammar_tool   = agent_tool.AgentTool(agent=grammar_agent)
quiz_tool      = agent_tool.AgentTool(agent=quiz_agent)
tts_tool       = agent_tool.AgentTool(agent=tts_agent)
split_tool     = agent_tool.AgentTool(agent=sentence_agent)

#pipeline_agent = LoopAgent(
#    name="learning_pipeline",
#    sub_agents=[retriever_agent, grammar_agent, quiz_agent],
#    max_iterations=1,
#)

#pipeline_tool = AgentTool(pipeline_agent)

router_agent = LlmAgent(
    model=MODEL,
    name="router",
    instruction=load_instruction_from_file("router_instruction_4.txt"),
    tools=[retriever_tool, grammar_tool, quiz_tool, tts_tool, split_tool,set_article_tool],
    output_key="router_reply"   # enables state writes
)

root_agent = router_agent


