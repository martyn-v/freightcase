from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal

WeightUnit = Literal["kg", "lb", "t", "g"]

_TO_KG: dict[WeightUnit, float] = {
    "kg": 1.0,
    "lb": 0.45359237,
    "t": 1000.0,  # metric tonne
    "g": 0.001,
}

# What the LLM is allowed to emit vs. what emails actually say
_WEIGHT_UNIT_ALIASES: dict[str, WeightUnit] = {
    "kg": "kg",
    "kgs": "kg",
    "kilo": "kg",
    "kilos": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "t": "t",
    "ton": "t",
    "tons": "t",
    "tonne": "t",
    "tonnes": "t",
    "mt": "t",
    "g": "g",
    "gr": "g",
    "gram": "g",
    "grams": "g",
}


class Incoterm(BaseModel):
    rule: Literal[
        "EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP"
    ]
    named_place: str


class Weight(BaseModel):
    """Weight as stated in the source document. `kg` is the canonical
    value; downstream code must only ever read `kg`."""

    value: float = Field(
        gt=0, description="Numeric value exactly as stated in the email"
    )
    unit: WeightUnit = Field(description="Unit as stated; common aliases are accepted")

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, v: str) -> WeightUnit:
        key = str(v).strip().lower().rstrip(".")
        if key not in _WEIGHT_UNIT_ALIASES:
            raise ValueError(
                f"Unrecognized weight unit '{v}'; expected one of {sorted(set(_WEIGHT_UNIT_ALIASES))}"
            )
        return _WEIGHT_UNIT_ALIASES[key]

    @computed_field  # appears in .model_dump(), so the MCP payload carries it
    @property
    def kg(self) -> float:
        return round(self.value * _TO_KG[self.unit], 3)


class CargoLine(BaseModel):
    description: str
    hs_code_hint: str | None
    pieces: int = Field(gt=0)
    weight: Weight
    # volume: Volume
    # dimensions: Dimensions | None
    # hazmat: HazmatInfo | None


class Location(BaseModel):
    locode: str = Field(
        min_length=5,
        max_length=5,
        description="The UN/LOCODE of the location",
        examples=["USNYC", "GBLON", "NLRTM", "SGSIN"],
    )


class QuoteRequest(BaseModel):
    mode: Literal["ocean_fcl", "ocean_lcl", "air", "rail", "road", "multimodal"]
    origin: Location
    destination: Location
    incoterm: Incoterm
    cargo: list[CargoLine] = Field(
        min_length=1, description="The cargo lines to be quoted"
    )
