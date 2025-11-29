from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from .util import load_instruction_from_file
from .fetch_and_store_article import fetch_and_store_article  # your new tool


MODEL = "gemini-2.5-flash-lite"

# --- Specialist agents ---

# 1) Article presenter: ONLY shows article
article_agent = LlmAgent(
    model=MODEL,
    name="article_presenter",
    instruction=(
    "You present only the article below.\n"
    "ARTICLE_JSON:\n"
    "{article?}\n\n"
    "Rules:\n"
    "- If ARTICLE_JSON is empty or missing a non-empty 'text', reply exactly: Give me a Yle URL first.\n"
    "- Otherwise print:\n"
    "  Article title: <title>\n"
    "  Article text:\n"
    "  <full text>\n"
    "- Do not add grammar, vocab, or quizzes."
	),
)

# 2) Grammar teacher
grammar_agent = LlmAgent(
    model=MODEL,
    name="selko_grammar_teacher",
    instruction=load_instruction_from_file("grammar_instruction.txt"),
    output_key="grammar_analyzed",
)

# 3) Quiz maker
quiz_agent = LlmAgent(
    model=MODEL,
    name="selko_quiz_maker",
    instruction=load_instruction_from_file("quiz_instruction.txt"),
    output_key="quiz_question",
)

# Wrap specialists as tools so router can call them conditionally
article_tool = AgentTool(article_agent)
grammar_tool = AgentTool(grammar_agent)
quiz_tool = AgentTool(quiz_agent)

# --- Root router agent (LLM-driven interactive flow) ---
router_agent = LlmAgent(
    model=MODEL,
    name="router",
    instruction=(
        "You are an interactive Selkouutiset study assistant.\n\n"

        "Available tools:\n"
        "1) fetch_and_store_article(user_text) -> dict {'title':..., 'text':...} or {'status':'no_url'}\n"
        "2) article_presenter\n"
        "3) selko_grammar_teacher\n"
        "4) selko_quiz_maker\n\n"

        "State keys you may rely on:\n"
        "- session.state['article'] : stored article (may be JSON text or dict)\n"
        "- session.state['choice'] : one of 'article', 'grammar', 'quiz'\n\n"

        "Workflow rules:\n"
        "A) First priority: if user provides a URL OR article is missing,\n"
        "   call fetch_and_store_article(user_text) exactly once.\n"
        "   - If tool returns status=no_url, ask user for a Yle URL and stop.\n"
        "   - If tool returns title/text, output ONLY that JSON to the user.\n"
        "     (This will be stored into session.state['article'] by ADK.)\n"
        "     Then ask: 'What do you want next? article / grammar / quiz' and stop.\n\n"

        "B) If an article already exists and user has not chosen yet,\n"
        "   ask: 'What do you want? article / grammar / quiz' and stop.\n\n"

        "C) If user replies with a choice:\n"
        "   - article -> call article_presenter only.\n"
        "   - grammar -> call selko_grammar_teacher only.\n"
        "   - quiz -> call selko_quiz_maker only.\n"
        "   Never call more than one specialist tool per turn.\n\n"

        "D) If user asks general questions like 'what can you do?', answer normally.\n"
        "   Explain that you can fetch a Yle selkouutiset article, show it, "
        "   teach grammar/vocab, or make quizzes.\n"
        
        "After you call any specialist tool (article_presenter / selko_grammar_teacher / selko_quiz_maker), you MUST return ONLY the tool’s response to the user and stop.\n"
		"Do NOT add any extra text such as “What do you want next?” in the same turn.\n"
		"Ask follow-up questions only on the next user turn.\n"
    ),
    tools=[fetch_and_store_article, article_tool, grammar_tool, quiz_tool],
)

root_agent = router_agent
