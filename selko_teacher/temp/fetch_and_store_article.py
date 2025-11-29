import re
from google.adk.tools import ToolContext
from .fetch_yle_article import get_article

URL_RE = re.compile(r"https?://\S+")

def fetch_and_store_article(user_text: str, tool_context: ToolContext) -> dict:
    """
    Finds a URL in user_text, fetches the Yle article, and stores it in session.state.
    Returns {"status": "no_url"} if no URL is found.
    """
    m = URL_RE.search(user_text or "")
    if not m:
        return {"status": "no_url"}

    url = m.group(0)
    data = get_article(url)  # your existing fetcher, returns {"title":..., "text":...}

    # Store deterministically
    tool_context.state["url"] = url
    tool_context.state["article"] = data

    # Reset choice / downstream outputs if they exist
    if "choice" in tool_context.state:
        del tool_context.state["choice"]
    if "grammar_analyzed" in tool_context.state:
        del tool_context.state["grammar_analyzed"]
    if "quiz_question" in tool_context.state:
        del tool_context.state["quiz_question"]

    return {"status": "ok", "title": data.get("title", "")}

