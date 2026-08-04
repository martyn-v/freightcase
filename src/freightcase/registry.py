from dataclasses import dataclass
from typing import Literal

from freightcase.specialists.base import SpecialistSchema
from freightcase.specialists.quote import QuoteRequest

Specialization = Literal["quote_request"]
Specialist = Literal["quote_specialist"]


@dataclass(frozen=True)
class RegistryEntry:
    description: str
    subgraph: Specialist
    tool: str  # the MCP tool the execute node calls — must exist on the TMS server
    schema: type[SpecialistSchema]


REGISTRY: dict[Specialization, RegistryEntry] = {
    "quote_request": RegistryEntry(
        description="Extracts and confirms quote requests from emails.",
        subgraph="quote_specialist",
        tool="create_quote",
        schema=QuoteRequest,
    )
}
