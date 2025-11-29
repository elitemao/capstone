from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import google_search, FunctionTool
from google.adk.tools import agent_tool
from .util import load_instruction_from_file
from .fetch_yle_article import get_article

import edge_tts

# -------------------------
# Config
# -------------------------

MODEL = "gemini-2.5-flash-lite"  # adjust if needed

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
    Saves the full path of the generated audio file in session.state["audio"].
    """
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(out_path)
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
    instruction=(
    	"At the start of every turn:\n"
    		"- If `state['greeted']` is not true, greet the user briefly and set `state['greeted']`=true."
			"- Otherwise do not greet again."

       	"You are the ONLY user-facing agent.\n"
        "All sub-agents are silent workers that write results to state.\n\n"
		
		"FORMATTING RULE:\n"
    		"- Whenever you show the menu, you MUST format all options in Markdown and in bold face"
    		"like **A) Show the news content**, **B) Learn vocabulary/grammar**, **C) Take a quiz**, **D) Listen sentence-by-sentence (TTS)**.\n\n"

        "Rules:\n"
        "1) If no article URL/content yet, ask for URL. STOP.\n"
        "2) Show the following menu:\n"
        	"A) Show the news content\n"
        	"B) Learn vocabulary/grammar\n"
        	"C) Take a quiz\n"
        	"D) Listen sentence-by-sentence (TTS)\n"
        "3) If user chooses A, call news_retriever, then read state['article'] "
        "and show it to user.\n"
        "4) If user chooses B, call selko_grammar_teacher, then read state['grammar_analyzed'].\n"
        "5) If user chooses C, call quiz_agent, then read state['quiz_question'].\n"
        "6) If user chooses D:\n"
        "   - call sentence_agent\n"
        "   - show numbered sentences from state['sentence_list']\n"
        "   - ask for a number and STOP.\n"
        "   - when number arrives, call tts_agent with that sentence\n"
        "   - show where audio file is saved.\n\n"

        "After responding to user's choice(A/B/C/D), ALWAYS end with the following section:\n\n"
        "**What would you like to do next?**"
        "** A) Show the news content**\n"
        "** B) Learn vocabulary/grammar**\n"
        "** C) Take a quiz**\n"
        "** D) Listen sentence-by-sentence (TTS)**\n"
        "Then STOP and wait."
    ),
    tools=[retriever_tool, grammar_tool, quiz_tool, tts_tool, split_tool],
    output_key="router_reply"   # enables state writes
)

root_agent = router_agent


