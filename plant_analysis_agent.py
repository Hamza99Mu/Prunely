from crewai import Agent

from tools import KnowledgeBaseSearchTool


def create_agent(
    llm,
    kb_tool: KnowledgeBaseSearchTool,
) -> Agent:

    return Agent(

        role="Plant Analysis Agent",

        goal=(
            "Analyze the user's text-only plant description "
            "and identify the most likely plant or plant group, "
            "while clearly separating confidence from uncertainty."
        ),

        backstory=(
            "You are a horticultural identification analyst. "
            "You work only from text supplied by the user and "
            "verified Prunely knowledge. You never pretend to see "
            "an image. You must search the Prunely Knowledge Base "
            "before making plant-specific claims."
        ),

        tools=[kb_tool],

        llm=llm,

        allow_delegation=False,

        verbose=False,

        max_iter=4,
    )
