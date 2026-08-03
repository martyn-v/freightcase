from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import Literal

WeightUnit = Literal["kg", "lb", "t", "g"]

_TO_KG: dict[str, float] = {
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
    # Spanish (LATAM emails are in scope)
    "tonelada": "t",
    "toneladas": "t",
    "kilogramo": "kg",
    "kilogramos": "kg",
    "libra": "lb",
    "libras": "lb",
    "gramo": "g",
    "gramos": "g",
}

DimensionUnit = Literal["cm", "m", "in", "ft"]

_TO_CM: dict[str, float] = {
    "cm": 1.0,
    "m": 100.0,
    "in": 2.54,
    "ft": 30.48,
}

_DIMENSION_UNIT_ALIASES: dict[str, DimensionUnit] = {
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    # Spanish (LATAM emails are in scope)
    "metro": "m",
    "metros": "m",
    "centímetro": "cm",
    "centímetros": "cm",
    "centimetro": "cm",
    "centimetros": "cm",
    "pulgada": "in",
    "pulgadas": "in",
    "pie": "ft",
    "pies": "ft",
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
    # `str`, not the WeightUnit enum: an enum in the prompt schema pressures the
    # model to convert ("toneladas" -> "t"), which is interpretation. The model
    # transcribes; normalize_unit maps to canonical or fails loudly (rule 1).
    unit: str = Field(description="Weight unit exactly as written in the email")

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


class Dimensions(BaseModel):
    """Dimensions as stated in the source document. `cm` is the canonical value; downstream code must only ever read `cm`. Converts to cubic meters for volume calculations."""

    length: float = Field(
        gt=0, description="Numeric value exactly as stated in the email"
    )
    width: float = Field(
        gt=0, description="Numeric value exactly as stated in the email"
    )
    height: float = Field(
        gt=0, description="Numeric value exactly as stated in the email"
    )
    # `str`, not the enum — same transcription rationale as Weight.unit.
    unit: str = Field(description="Dimension unit exactly as written in the email")

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, v: str) -> DimensionUnit:
        key = str(v).strip().lower().rstrip(".")
        if key not in _DIMENSION_UNIT_ALIASES:
            raise ValueError(
                f"Unrecognized dimension unit '{v}'; expected one of {sorted(set(_DIMENSION_UNIT_ALIASES))}"
            )
        return _DIMENSION_UNIT_ALIASES[key]

    @computed_field
    @property
    def length_cm(self) -> float:
        return round(self.length * _TO_CM[self.unit], 3)

    @computed_field
    @property
    def width_cm(self) -> float:
        return round(self.width * _TO_CM[self.unit], 3)

    @computed_field
    @property
    def height_cm(self) -> float:
        return round(self.height * _TO_CM[self.unit], 3)

    @computed_field
    @property
    def volume_m3(self) -> float:
        """Volume in cubic meters, rounded to 3 decimal places."""
        return round(
            (self.length_cm / 100) * (self.width_cm / 100) * (self.height_cm / 100), 3
        )


class CargoLine(BaseModel):
    """Absence is degraded, not fatal (rule 3): unstated pieces/weight/dims
    extract as None and surface via missing_for_quoting() for HITL remedy.
    Schema validation polices form (bad units, negative values), not absence."""

    description: str
    hs_code_hint: str | None = None
    pieces: int | None = Field(default=None, gt=0)
    weight: Weight | None = None
    dimensions: Dimensions | None = None


class Location(BaseModel):
    """A location as evidenced in the email. Capture what is stated;
    resolution to a canonical LOCODE is downstream, not the model's job."""

    name: str | None = Field(
        default=None,
        description="Location name exactly as written in the email, e.g. 'Cartagena, Colombia'",
    )
    locode: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{2}[A-Z2-9]{3}$",
        description="UN/LOCODE, only if explicitly stated in the email",
        examples=["USNYC", "NLRTM"],
    )
    iata: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{3}$",
        description="IATA airport code, only if explicitly stated in the email, e.g. 'BOG'",
    )

    @field_validator("locode", "iata", mode="before")
    @classmethod
    def uppercase(cls, v):
        return v.upper() if isinstance(v, str) else v

    @model_validator(mode="after")
    def at_least_one(self) -> "Location":
        if not (self.name or self.locode or self.iata):
            raise ValueError("Location requires at least one of: name, locode, iata")
        return self


class QuoteRequest(BaseModel):
    mode: (
        Literal["ocean_fcl", "ocean_lcl", "air", "rail", "road", "multimodal"] | None
    ) = None
    origin: Location
    destination: Location
    incoterm: Incoterm | None = None
    cargo: list[CargoLine] = Field(
        min_length=1, description="The cargo lines to be quoted"
    )

    def missing_for_quoting(self) -> list[str]:
        """Deterministic completeness check: field paths that are absent from
        the email but needed to produce a quote. Distinct from schema validity —
        a request can be valid yet not quotable. Feeds HITL per-field flags."""
        missing: list[str] = []
        if self.mode is None:
            missing.append("mode")
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


