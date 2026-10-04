from crewai import Agent

from tools import KnowledgeBaseSearchTool


def create_agent(
    llm,
    kb_tool: KnowledgeBaseSearchTool,
) -> Agent:

    return Agent(

        role="Data Verifier Agent",

        goal=(
            "Audit the analysis and pruning plan against "
            "the source workbook, remove unsupported claims, "
            "and produce a traceable final answer."
        ),

        backstory=(
            "You are the quality-control gate for Prunely. "
            "You distrust unsupported horticultural claims. "
            "You search the knowledge base again, compare the "
            "previous agents' claims with source records, and "
            "explicitly flag uncertainty or missing evidence."
        ),

        tools=[kb_tool],

        llm=llm,

        allow_delegation=False,

        verbose=False,

        max_iter=5,
    )
