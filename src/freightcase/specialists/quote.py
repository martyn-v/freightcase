from typing import Literal

from pydantic import Field

from freightcase.specialists.base import SpecialistSchema
from freightcase.specialists.common import CargoLine, Incoterm, Location


class QuoteRequest(SpecialistSchema):
    mode: (
        Literal["ocean_fcl", "ocean_lcl", "air", "rail", "road", "multimodal"] | None
    ) = None
    origin: Location
    destination: Location
    incoterm: Incoterm | None = None
    cargo: list[CargoLine] = Field(
        min_length=1, description="The cargo lines to be quoted"
    )

    def missing_for_execution(self) -> list[str]:
        """Deterministic completeness check: field paths that are absent from
        the email but needed to produce a quote. Distinct from schema validity —
        a request can be valid yet not quotable. Feeds HITL per-field flags."""
        missing: list[str] = []
        if self.mode is None:
            missing.append("mode")
        # The TMS write needs a LOCODE specifically; a name alone can't
        # execute, and a stated IATA code is informative context for whoever
        # supplies the LOCODE at the gate (or a future auto-resolver), never
        # a substitute. Resolution is downstream of extraction by design.
        if self.origin.locode is None:
            missing.append("origin.locode")
        if self.destination.locode is None:
            missing.append("destination.locode")
        if self.incoterm is None:
            missing.append("incoterm")
        for i, line in enumerate(self.cargo):
            if line.pieces is None:
                missing.append(f"cargo.{i}.pieces")
            if line.weight is None:
                missing.append(f"cargo.{i}.weight")
            # FCL is priced per container; dims implied. Unknown mode: be
            # conservative and ask for dims.
            if self.mode != "ocean_fcl" and line.dimensions is None:
                missing.append(f"cargo.{i}.dimensions")
        return missing

    def summarize(self) -> str:
        """One sentence stating what confirming will execute. Composed here so
        every surface shows the same truth; totals use canonical kg. Unstated
        fields are named as gaps, never omitted — the human must see them."""
        mode = self.mode or "mode not stated"
        stated_pieces = [line.pieces for line in self.cargo if line.pieces is not None]
        pieces = (
            f"{sum(stated_pieces)} pieces" if stated_pieces else "pieces not stated"
        )
        stated_kg = [line.weight.kg for line in self.cargo if line.weight is not None]
        kg = f"{sum(stated_kg):g} kg" if stated_kg else "weight not stated"
        incoterm = (
            f"{self.incoterm.rule} {self.incoterm.named_place}"
            if self.incoterm is not None
            else "incoterm not stated"
        )
        return (
            f"Create a quote request: {mode}, "
            f"{self.origin.label()} → {self.destination.label()}, "
            f"{pieces}, {kg}, {incoterm}."
        )
