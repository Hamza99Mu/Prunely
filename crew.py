from crewai import Crew, Process, Task

from agents.data_verifier_agent import (
    create_agent as create_verifier_agent,
)

from agents.plant_analysis_agent import (
    create_agent as create_analysis_agent,
)

from agents.pruning_expert_agent import (
    create_agent as create_pruning_agent,
)

from knowledge_base import KnowledgeBase
from llm import build_llm
from tools import KnowledgeBaseSearchTool


def run_prunely(
    user_query: str,
    kb: KnowledgeBase,
) -> dict:

    llm = build_llm()

    analysis_tool = (
        KnowledgeBaseSearchTool(kb=kb)
    )

    pruning_tool = (
        KnowledgeBaseSearchTool(kb=kb)
    )

    verifier_tool = (
        KnowledgeBaseSearchTool(kb=kb)
    )

    plant_agent = create_analysis_agent(
        llm,
        analysis_tool,
    )

    pruning_agent = create_pruning_agent(
        llm,
        pruning_tool,
    )

    verifier_agent = create_verifier_agent(
        llm,
        verifier_tool,
    )

    analysis_task = Task(

        description=f"""
User's text-only plant request:

{user_query}

First, use the "Prunely Knowledge Base Search" tool.

Search for the plant and its identifying characteristics.

Then produce:

1. Most likely plant/common name
2. Scientific name if supported
3. Confidence: High / Medium / Low
4. Evidence from the knowledge base
5. What cannot be confirmed from text alone

Do not claim to have seen an image.
Do not invent plant-specific facts.
""",

        expected_output=(
            "A concise, evidence-based plant analysis."
        ),

        agent=plant_agent,
    )

    pruning_task = Task(

        description=f"""
Create a pruning plan for this user's request:

{user_query}

You will receive the Plant Analysis Agent's result
as context.

You MUST use the "Prunely Knowledge Base Search" tool again.

Search specifically for pruning rules for the identified
plant or the closest verified plant record.

Produce:

- Recommended pruning type/purpose
- Best pruning period
- Periods/conditions to avoid
- Growth stage
- Branches to remove
- Branches to retain
- Where to cut
- Tools required
- Step-by-step pruning sequence
- Maximum removal, only if the source explicitly supports it
- Aftercare
- Special notes
- Safety/uncertainty notes

If a requested fact is not in the source, say:

"Not specified in the source"

instead of guessing.
""",

        expected_output=(
            "A practical pruning plan grounded in the source."
        ),

        agent=pruning_agent,
    )

    verifier_task = Task(

        description=f"""
Audit the complete Prunely answer for this user request:

{user_query}

You will receive the Plant Analysis Agent and
Pruning Expert Agent outputs as context.

You MUST use the "Prunely Knowledge Base Search"
tool again.

Verify important plant identity and pruning claims
against the source records.

Return the final answer in this structure:

# Prunely Recommendation

## Plant

## Confidence

## Pruning Plan

## What to Avoid

## Aftercare

## Safety & Uncertainty

## Source Traceability

For Source Traceability, list the relevant source
sheet names, row numbers, plant IDs, and source URL
available from the tool results.

Rules:

- Remove or soften unsupported claims.
- Never fabricate missing data.
- If the source does not establish an exact date,
  do not invent one.
- If the plant cannot be confidently identified from
  text, say so and give only conditional guidance
  supported by the source.
- Keep the final response practical for an ordinary
  home gardener.
""",

        expected_output=(
            "A verified final pruning recommendation "
            "with traceability."
        ),

        agent=verifier_agent,
    )

    crew = Crew(

        agents=[
            plant_agent,
            pruning_agent,
            verifier_agent,
        ],

        tasks=[
            analysis_task,
            pruning_task,
            verifier_task,
        ],

        process=Process.sequential,

        verbose=False,
    )

    result = crew.kickoff()

    outputs = result.tasks_output

    return {

        "final": (
            result.raw
            if hasattr(result, "raw")
            else str(result)
        ),

        "analysis": (
            outputs[0].raw
            if len(outputs) > 0
            else ""
        ),

        "pruning": (
            outputs[1].raw
            if len(outputs) > 1
            else ""
        ),

        "verification": (
            outputs[2].raw
            if len(outputs) > 2
            else ""
        ),
    }
