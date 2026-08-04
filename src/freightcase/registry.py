from typing import Literal

from pydantic import BaseModel

Specialization = Literal["quote_request"]
Specialist = Literal["quote_specialist"]


class RegistryEntry(BaseModel):
    description: str
    graph: Specialist


REGISTRY: dict[Specialization, RegistryEntry] = {
    "quote_request": RegistryEntry(
        description="Extracts and confirms quote requests from emails.",
        graph="quote_specialist",
    )
}
