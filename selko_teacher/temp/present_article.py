def present_article(**kwargs) -> str:
    # ADK passes the context here, but we don't type it in the signature
    tool_context = kwargs.get("tool_context") or kwargs.get("ctx")
    state = getattr(tool_context, "state", None) if tool_context else None

    if not state:
        return "Give me a Yle URL first."

    article = state.get("article")
    if not article or not article.get("text"):
        return "Give me a Yle URL first."

    title = article.get("title", "")
    text = article.get("text", "")
    return f"Article title: {title}\n\nArticle text:\n{text}"
