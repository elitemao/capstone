from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from .util import load_instruction_from_file
from .fetch_yle_article import get_article
# -------------------------
# Config
# -------------------------

MODEL = "gemini-2.5-flash-lite"  # adjust if needed


# -------------------------
# Internal agents: analyst, grammar, quiz
# -------------------------

# 1) Analyst: uses the RSS tool and outputs JSON only
retriever_agent = LlmAgent(
    model=MODEL,
    name="news_retriever",
    instruction=load_instruction_from_file("news_extraction_instruction.txt"),
    tools=[get_article],
    output_key="article",
)





# 2) Grammar agent: receives that JSON and explains grammar in Markdown
grammar_agent = LlmAgent(
    model=MODEL,
    name="selko_grammar_teacher",
    instruction=load_instruction_from_file("grammar_instruction.txt"),
    tools=[],
    output_key="grammar_analyzed"
)

# 3) Quiz agent: receives the same JSON and creates a quiz in Markdown
quiz_agent = LlmAgent(
    model=MODEL,
    name="selko_quiz_maker",
    instruction=load_instruction_from_file("quiz_instruction.txt"),
    tools=[],
    output_key="quiz_question"
)


pipeline_agent = LoopAgent(
    name="learning_pipeline",
    sub_agents=[retriever_agent, grammar_agent, quiz_agent],
    max_iterations=1,
)

#pipeline_tool = AgentTool(pipeline_agent)

#router_agent = LlmAgent(
#    model=MODEL,
#    name="router",
#    instruction=(
#        "You are the front-door assistant. "
#        "Answer normal user questions directly (e.g., 'what can you do?', greetings, help). "
#        "ONLY continue the Finnish lession when the user provides a Yle URL "
#        "or clearly asks to study a specific Yle selkouutiset article. "
#        "If no URL is given but the user wants selkouutiset study, ask for the URL."
#        "After running the 'pipeline_agent' ask the user if they want to see the news content, to learn the vocabularies, to learn the grammar or to have quiz."
#        #"Ask the user if they want to see the news content, to learn the vocabularies, to learn the grammar or to have quiz repeatedly."
#        "If user wants to see the news content, you run 'retriever_agent'"
#        "If user wants the vocabulary or grammar, you have to run 'retriever_agent' first and then 'grammar_agent'"
#        "If user wants quiz, you have to run 'retriever_agent' first and then 'quiz_agent'" 
#        "After responding to the user, ask again if the user want to see the news content, to learn the vocabularies, to learn the grammar or to have quiz"
#    ),
#    tools=[pipeline_tool],
#    #sub_agents=[retriever_agent, grammar_agent, quiz_agent]
#)

root_agent = pipeline_agent


