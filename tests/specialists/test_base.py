from freightcase.specialists.quote import QuoteRequest


class TestConfidence:
    def test_provenance_per_field(self):
        """The transcription proof: the model said 'toneladas', the validator
        made it 't' — confidence() must expose that the value was normalized,
        while untouched fields are 'stated' and absent ones 'missing'."""
        raw = {
            "mode": "road",
            "origin": {"name": "Bogota"},
            "destination": {"name": "Medellin"},
            "incoterm": {"rule": "DAP", "named_place": "Medellin"},
            "cargo": [
                {
                    "description": "Packed foodstuffs",
                    "pieces": 12,
                    "weight": {"value": 8.4, "unit": "toneladas"},
                    "dimensions": {
                        "length": 120,
                        "width": 80,
                        "height": 60,
                        "unit": "cm",
                    },
                },
                {"description": "Machine parts"},
            ],
        }
        qr = QuoteRequest.model_validate(raw)
        assert qr.cargo[0].weight is not None
        assert qr.cargo[0].weight.unit == "t"  # code normalized, model didn't

        conf = qr.confidence(raw)

        # Normalization is reported at the leaf that changed, not the container.
        assert conf["cargo.0.weight.unit"] == "normalized"  # toneladas -> t
        assert conf["cargo.0.weight.value"] == "stated"
        assert conf["cargo.0.dimensions.unit"] == "stated"  # cm passed through
        assert conf["cargo.0.dimensions.length"] == "stated"
        assert conf["mode"] == "stated"
        assert conf["incoterm.rule"] == "stated"
        assert conf["incoterm.named_place"] == "stated"
        assert conf["origin.name"] == "stated"
        assert conf["origin.locode"] == "missing"  # not stated in the email
        assert conf["cargo.0.pieces"] == "stated"
        assert conf["cargo.0.description"] == "stated"
        assert conf["cargo.0.hs_code_hint"] == "missing"
        # None containers are reported at container level; no leaves beneath.
        assert conf["cargo.1.pieces"] == "missing"
        assert conf["cargo.1.weight"] == "missing"
        assert conf["cargo.1.dimensions"] == "missing"
        assert "cargo.1.weight.unit" not in conf

    def test_uppercased_locode_reports_normalized(self):
        raw = {
            "origin": {"name": "Shanghai", "locode": "cnsha"},
            "destination": {"name": "Cartagena"},
            "cargo": [{"description": "Tiles"}],
        }
        qr = QuoteRequest.model_validate(raw)

        conf = qr.confidence(raw)

        assert conf["origin.locode"] == "normalized"  # cnsha -> CNSHA
        assert conf["origin.name"] == "stated"
