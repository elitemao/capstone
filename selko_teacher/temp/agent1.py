import os
import json
import requests
import feedparser

from markdown import markdown as md_to_html

from google.genai.types import Content, Part

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool


# -------------------------
# Config
# -------------------------

MODEL = "gemini-2.5-flash-lite"  # adjust if needed
APP_NAME = "selko_finnish_classroom"

# Don't crash if GOOGLE_API_KEY is missing – ADK Web will handle env loading.
if "GOOGLE_API_KEY" not in os.environ:
    print("WARNING: GOOGLE_API_KEY not set in environment.")


# -------------------------
# Tool: fetch latest Selkouutiset via RSS
# -------------------------

SELKOUUTISET_RSS = (
    "https://feeds.yle.fi/uutiset/v1/recent.rss?publisherIds=YLE_SELKOUUTISET"
)

def fetch_latest_selkouutiset(n: int = 1):
    """
    Fetch the latest n Selkouutiset items from Yle RSS feed.
    Returns a list of dicts: {title, summary, link}.
    """
    resp = requests.get(SELKOUUTISET_RSS, timeout=10)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    items = []
    for entry in feed.entries[:n]:
        items.append(
            {
                "title": entry.get("title", "") or "",
                "summary": entry.get("summary", "")
                or entry.get("description", "")
                or "",
                "link": entry.get("link", "") or "",
            }
        )
    return items


# -------------------------
# Internal agents: analyst, grammar, quiz
# -------------------------

# 1) Analyst: uses the RSS tool and outputs JSON only
news_retriever_agent = LlmAgent(
    model=MODEL,
    name="news_retriever",
    instruction=load_instruction_from_file("news_retriever_instruction.txt"),
    tools=[fetch_yle_article],
    output_key=article,
)





# 2) Grammar agent: receives that JSON and explains grammar in Markdown
grammar_agent = LlmAgent(
    model=MODEL,
    name="selko_grammar_teacher",
    instruction=load_instruction_from_file("grammar_instruction.txt"),
    tools=[],
    output_key=grammar_analyzed
)

# 3) Quiz agent: receives the same JSON and creates a quiz in Markdown
quiz_agent = LlmAgent(
    model=MODEL,
    name="selko_quiz_maker",
    instruction=load_instruction_from_file("quiz_instruction.txt"),
    tools=[],
)


# -------------------------
# Session service + runners
# -------------------------

session_service = InMemorySessionService()

analyst_runner = Runner(
    agent=analyst_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

grammar_runner = Runner(
    agent=grammar_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

quiz_runner = Runner(
    agent=quiz_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


async def ensure_session(session_id: str):
    """Get or create a session for a given id."""
    session = await session_service.get_session(
        user_id=session_id,
        app_name=APP_NAME,
        session_id=session_id,
    )
    if session is None:
        await session_service.create_session(
            user_id=session_id,
            app_name=APP_NAME,
            session_id=session_id,
        )


# -------------------------
# Orchestrator: build one Finnish lesson as HTML
# -------------------------

async def build_finnish_lesson():
    """
    Multi-step pipeline:
      1) analyst_runner → JSON
      2) grammar_runner → Markdown
      3) quiz_runner → Markdown
      4) combine into HTML page
    Returns an HTML string.
    """
    analyst_session = "selko_analyst"
    grammar_session = "selko_grammar"
    quiz_session = "selko_quiz"

    await ensure_session(analyst_session)
    await ensure_session(grammar_session)
    await ensure_session(quiz_session)

    # 1) ANALYST → JSON
    analyst_prompt = Content(
        role="user",
        parts=[Part(text="Analyze the latest Selkouutiset article and output JSON as instructed.")],
    )

    analyst_json_str = ""

    async for event in analyst_runner.run_async(
        user_id=analyst_session,
        session_id=analyst_session,
        new_message=analyst_prompt,
    ):
        # Grab ANY text chunks from the model, not only "final" ones.
        if getattr(event, "content", None) and event.content.parts:
            chunk = "".join(
                part.text
                for part in event.content.parts
                if getattr(part, "text", None)
            )
            if chunk.strip():
                analyst_json_str = chunk  # keep the latest non-empty text


    if not analyst_json_str:
        return "<html><body><h1>Error</h1><p>No JSON from analyst agent.</p></body></html>"

    # --- Clean up possible markdown fences etc. ---
    raw = analyst_json_str.strip()

    # If wrapped in ```...``` or ```json ... ```
    if raw.startswith("```"):
        lines = raw.splitlines()
        # drop first fence line
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # drop last fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # If it starts with "json" (from ```json)
    if raw.lower().startswith("json"):
        raw = raw[4:].lstrip()

    # If there is text before the first '{', cut everything before it
    first_brace = raw.find("{")
    if first_brace != -1:
        raw = raw[first_brace:]

    try:
        analyst_data = json.loads(raw)
    except json.JSONDecodeError:
        escaped = raw.replace("<", "&lt;").replace(">", "&gt;")
        return (
            "<html><body><h1>JSON parse error</h1>"
            f"<pre>{escaped}</pre></body></html>"
        )


    json_for_others = json.dumps(analyst_data, ensure_ascii=False)

    # 2) GRAMMAR → Markdown
    grammar_prompt = Content(
        role="user",
        parts=[Part(text=json_for_others)],
    )

    grammar_text = ""
    async for event in grammar_runner.run_async(
        user_id=grammar_session,
        session_id=grammar_session,
        new_message=grammar_prompt,
    ):
        if getattr(event, "content", None) and event.content.parts:
            chunk = "".join(
                part.text
                for part in event.content.parts
                if getattr(part, "text", None)
            )
            if chunk.strip():
                grammar_text = chunk


    # 3) QUIZ → Markdown
    quiz_prompt = Content(
        role="user",
        parts=[Part(text=json_for_others)],
    )

    quiz_text = ""
    async for event in quiz_runner.run_async(
        user_id=quiz_session,
        session_id=quiz_session,
        new_message=quiz_prompt,
    ):
        if getattr(event, "content", None) and event.content.parts:
            chunk = "".join(
                part.text
                for part in event.content.parts
                if getattr(part, "text", None)
            )
            if chunk.strip():
                quiz_text = chunk


    # 4) Combine everything into one lesson
    title = analyst_data.get("title", "Yle Selkouutiset")
    link = analyst_data.get("link", "")
    summary_fi = analyst_data.get("summary_fi", "")

    grammar_html = md_to_html(grammar_text)
    quiz_html = md_to_html(quiz_text)
    
    html = f"""
<html>
  <head><meta charset="utf-8"><title>{title}</title></head>
  <body>
    <h1>Finnish Classroom Lesson</h1>

    <h2>Article</h2>
    <p><strong>{title}</strong><br>
       <a href="{link}" target="_blank" rel="noopener noreferrer">{link}</a></p>

    <h3>Finnish summary</h3>
    <p>{summary_fi}</p>

    <h2>Grammar Explanation</h2>
    {grammar_html}

    <h2>Quiz</h2>
    {quiz_html}
  </body>
</html>
""".strip()

    # ✨ NEW: save to HTML file in current directory
    filename = "selko_lesson.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    # For ADK Web: just tell the user where the file is
    return f"Saved Finnish lesson to **{filename}** in the server working directory.\n" \
       f"Open that file in your browser to view the full HTML lesson."


# Wrap orchestrator as a tool (ADK 1.18 style)
finnish_lesson_tool = FunctionTool(build_finnish_lesson)


# -------------------------
# Front-facing web agent
# -------------------------

web_instruction = (
    "You are the Selkouutiset Finnish Classroom web agent.\n"
    "When the user asks for a Finnish lesson, you MUST call the tool "
    "\"build_finnish_lesson\".\n"
    "Return ONLY the HTML produced by that tool."
)

web_agent = LlmAgent(
    model=MODEL,
    name="selko_teacher",
    description="you are Finnish learning helper, who reads in Finnish news fro yle.fi and lists vocalbularies and grammer in the new. Then you also create a quiz to the user",
    instruction=load_instruction_from_file("root_instruction.txt"),
    #tools=[finnish_lesson_tool],
    sub_agents=[analyst_agent,grammar_agent,quiz_agent]
)


# -------------------------
# root_agent for ADK Web
# -------------------------

root_agent = web_agent

# -------------------------
# CLI entry point (for running as a normal script)
# -------------------------

if __name__ == "__main__":
    import asyncio

    # Make sure API key is set when running from CLI
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("Please set GOOGLE_API_KEY in your environment.")

    # Run the pipeline once and print the message
    result = asyncio.run(build_finnish_lesson())
    print(result)

