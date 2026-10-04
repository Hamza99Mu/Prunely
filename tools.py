from typing import Any

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from knowledge_base import KnowledgeBase


class KnowledgeSearchInput(BaseModel):

    query: str = Field(
        ...,
        description=(
            "A focused plant/pruning knowledge query. "
            "Include the plant name and exact fact "
            "you need to verify."
        ),
    )


class KnowledgeBaseSearchTool(BaseTool):

    name: str = (
        "Prunely Knowledge Base Search"
    )

    description: str = (
        "Search the verified Prunely plant-pruning "
        "workbook using semantic vector retrieval. "
        "Returns source text plus sheet/row/plant "
        "metadata. Use this before making any "
        "plant-specific or pruning-specific claim."
    )

    args_schema: type[BaseModel] = (
        KnowledgeSearchInput
    )

    kb: Any

    def _run(
        self,
        query: str,
    ) -> str:

        return self.kb.format_search_results(
            query
        )
