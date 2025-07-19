import re
from datetime import datetime
from enum import Enum, IntEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class Color(IntEnum):
    Green = 1
    Yellow = 2
    Red = 3


class ClinicalDecisionRule(Enum):
    TreatmentRecommendation = "Treatment Recommendation"
    DiagnosisEvaluation = "Diagnosis Evaluation"
    ClinicalNotes = "Clinical Notes"
    VitalsChiefComplaintEvaluation = "Vitals & Chief Complaint Evaluation"
    InvestigationRecommendations = "Investigation Recommendations"


def global_validate_severity(v):
    if isinstance(v, str):
        try:
            return Color[v]
        except KeyError as e:
            raise ValueError(f"Invalid severity: {v}") from e


class ResponseValue(BaseModel):
    severity: Color = Field(validation_alias="Severity")
    reason: str = Field(validation_alias="Reason")

    @field_validator("severity", mode="before")
    def validate_severity(cls, v):
        return global_validate_severity(v)


class RecommendationValue(BaseModel):
    severity: Color = Field(validation_alias="Severity")
    action: str = Field(validation_alias="Action")

    @field_validator("severity", mode="before")
    def validate_severity(cls, v):
        return global_validate_severity(v)


class AIResponse(BaseModel):
    responses: list[ResponseValue] = Field(validation_alias="Response")
    recommendations: list[RecommendationValue] = Field(validation_alias="Recommendations")

    @model_validator(mode="after")
    def check_nonempty(cls, values):
        if not values.responses and not values.recommendations:
            raise ValueError(
                "AIResponse must have at least one response value or at least one recommendation value"
            )
        return values


class AICall(BaseModel):
    rule: ClinicalDecisionRule
    response: AIResponse
    user_id: str
    time: datetime
    thumbs_up_down: Literal["Up", "Down", "None"]
    user_role_prompt: str
    silent: Literal["Active", "Silent"]
    acknowledged: bool | None = None

    @property
    def color(self) -> Color:
        """We define the color of an AICall as the worst severity of the response or recommendation"""
        response_colors = {r.severity for r in self.response.responses}
        recommendation_colors = {r.severity for r in self.response.recommendations}
        all_colors = response_colors | recommendation_colors
        return max(all_colors)


class AICalls(BaseModel):
    calls: list[AICall]
    rule: ClinicalDecisionRule | None = None

    def _is_empty(self) -> bool:
        return not self.calls

    def for_rule(self, rule: ClinicalDecisionRule) -> "AICalls":
        relevant_calls = [call for call in self.calls if call.rule == rule]
        return AICalls(calls=relevant_calls, rule=rule)

    def for_rules(self, rules: list[ClinicalDecisionRule]) -> "AICalls":
        relevant_calls = [call for call in self.calls if call.rule in rules]
        return AICalls(calls=relevant_calls, rule=None)

    @property
    def final(self) -> AICall | None:
        if self.rule is None:
            raise ValueError("The final call is not defined for an AICalls object with no rule")
        if self._is_empty():
            return None
        return max(self.calls, key=lambda c: c.time)

    @property
    def first(self) -> AICall | None:
        if self.rule is None:
            raise ValueError("The first call is not defined for an AICalls object with no rule")
        if self._is_empty():
            return None
        return min(self.calls, key=lambda c: c.time)

    @property
    def final_color(self) -> Color | None:
        final = self.final
        return final.color if final else None

    @property
    def first_color(self) -> Color | None:
        first = self.first
        return first.color if first else None

    def any_final_color(self, color: Color) -> bool:
        final_colors_for_rules = {self.for_rule(rule).final_color for rule in ClinicalDecisionRule}
        return color in final_colors_for_rules

    def any_first_color(self, color: Color) -> bool:
        first_colors_for_rules = {self.for_rule(rule).first_color for rule in ClinicalDecisionRule}
        return color in first_colors_for_rules

    @property
    def colors_seen(self) -> set[Color] | None:
        if self._is_empty():
            return None
        else:
            return {call.color for call in self.calls}

    @property
    def worst_color(self) -> Color | None:
        colors = self.colors_seen
        return max(colors) if colors else None

    def ever_had_color(self, color: Color) -> bool | None:
        colors = self.colors_seen
        return color in colors if colors else None

    def final_is_color(self, color: Color) -> bool | None:
        fc = self.final_color
        return fc == color if fc else None

    def first_is_color(self, color: Color) -> bool | None:
        fc = self.first_color
        return fc == color if fc else None

    # convenience properties
    @property
    def final_red(self) -> bool | None:
        return self.final_is_color(Color.Red)

    @property
    def first_red(self) -> bool | None:
        return self.first_is_color(Color.Red)

    @property
    def final_red_yellow(self) -> bool | None:
        return self.final_is_color(Color.Red) or self.final_is_color(Color.Yellow)

    @property
    def ever_red(self) -> bool | None:
        return self.ever_had_color(Color.Red)

    @property
    def ever_red_yellow(self) -> bool | None:
        return self.ever_had_color(Color.Red) or self.ever_had_color(Color.Yellow)

    @property
    def any_final_red(self) -> bool | None:
        return self.any_final_color(Color.Red)

    @property
    def any_first_red(self) -> bool | None:
        return self.any_first_color(Color.Red)

    @property
    def any_final_red_yellow(self) -> bool | None:
        return self.any_final_color(Color.Red) or self.any_final_color(Color.Yellow)


def format_not_recorded(template: str, entry: BaseModel) -> str:
    entries = entry.model_dump()
    for k, v in entries.items():
        if v is None:
            entries[k] = "Not recorded"

    return template.format(**entries).strip()


HISTORY_TEMPLATE = """**Age:** {age}
**Gender:** {gender}

**Allergies:** {allergies}
**Social history:** {social_history}
**History of chronic illness:** {chronic_illness}
**Obstetric history:** {obs}

**Vitals:**
**Height:** {doc_hgt}
**Weight:** {doc_wgt}
**Heart rate:** {doc_bpm}
**Blood pressure:** {doc_bpr}
**Mean upper arm circumference:** {doc_muc}
**Temperature:** {doc_tmp}
**Respiratory rate:** {doc_rr}
**SpO2:** {doc_sp2}

**Chief complaint:** {cc}

**Clinical notes:**
{clinical_notes_clean}

**Structured examinations recorded:** {examination}"""

INVESTIGATION_TEMPLATE = """**Investigations conducted:**
{lab_test_clean}"""

DIAGNOSIS_TEMPLATE = """**Diagnoses:**
{dx}"""

TREATMENT_TEMPLATE = """**Referrals:**
{referrals}

**Medications:**
{rx}"""


INVESTIGATION_SUBFIELD_SORT_ORDER = {
    "Full Haemogram (FHG)": [
        "Result..",
        "WBC",
        "HGB",
        "HCT",
        "Plt",
        "RBC (Full Haemogram)",
        "MCV",
        "MCH",
        "MCHC",
        "RDW-CV",
        "RDW-SD",
        "Neutrophill percentage",
        "Mid-granulocyte Percentage",
        "Lymphocyte Percentage",
        "Neutrophill count",
        "Mid-granulocytes Count",
        "Lymphocytes Count",
        "MPV",
        "PDW",
        "PCT",
        "P-LCR",
        "P-LCC",
    ],
    "Urine Analysis": [
        "Result..",
        "Colour.",
        "Appearance (Urine Analysis)",
        "pH (Dipstick)",
        "Specific Gravity",
        "Glucose",
        "Ketones",
        "Proteins",
        "Blood (Dipstick)",
        "Leukocytes",
        "Nitrate",
        "Urobilinogen",
        "Pus cells",
        "RBC's (Urinalysis-Microscopy)",
        "Epithelial Cells",
        "Crystals -- Amount (Microscopy)",
        "Parasites (Urine Microscopy)",
        "Trichomonads (Urinalysis-Microscopy)",
        "Yeast cells (Microscopy)",
        "Amount of yeast cells",
        "Bilirubin.",
        "Casts - type",
        "Crystals -- Type",
    ],
    "Stool Microscopy": [
        "Result..",
        "Colour",
        "Consistency",
        "Blood (Gross Appearance)",
        "Mucous",
        "RBC's (Microscopy)",
        "Pus cells",
        "Parasites",
        "Yeast cells (Microscopy)",
        "Amount of yeast cells",
        "Crystals -- Amount",
        "Crystals -- Type",
    ],
}


class ClinicalDocumentation(BaseModel):
    gender: str = Field(validation_alias="Gender")
    age: str = Field(validation_alias="Age")
    allergies: str | None = Field(validation_alias="Allergies", default=None)
    cc: str | None = Field(validation_alias="CC", default=None)
    chronic_illness: str | None = Field(validation_alias="ChronicIllness", default=None)
    clinical_notes: str | None = Field(validation_alias="ClinicalNotes", default=None)
    clinical_notes_clean: str | None = Field(validation_alias="ClinicalNotes_clean", default=None)
    dx: str | None = Field(validation_alias="Dx", default=None)
    examination: str | None = Field(validation_alias="Examination", default=None)
    lab_test: str | None = Field(validation_alias="LabTest", default=None)
    obs: str | None = Field(validation_alias="obs", default=None)
    referrals: str | None = Field(validation_alias="Referrals", default=None)
    rx: str | None = Field(validation_alias="Rx", default=None)
    social_history: str | None = Field(validation_alias="SocialHistory", default=None)
    doc_bpr: str | None = Field(validation_alias="doc_bpr", default=None)
    doc_hgt: float | None = Field(validation_alias="doc_hgt", default=None)
    doc_muc: str | None = Field(validation_alias="doc_muc", default=None)
    doc_bpm: float | None = Field(validation_alias="doc_bpm", default=None)
    doc_tmp: float | None = Field(validation_alias="doc_tmp", default=None)
    doc_wgt: float | None = Field(validation_alias="doc_wgt", default=None)
    doc_rr: float | None = Field(validation_alias="doc_rr", default=None)
    doc_sp2: float | None = Field(validation_alias="doc_sp2", default=None)

    @computed_field
    @property
    def lab_test_clean(self) -> str | None:
        s = self.lab_test
        if s is None:
            return None

        if s == "Not recorded":
            return "Not recorded"

        s = s.replace("**Investigations conducted:**\n", "")
        s = s.replace("::", ":")
        investigation_list = s.split("\n\n")

        formatted_string = ""

        for investigation in investigation_list:
            investigation_name, investigation_value = investigation.split(":", maxsplit=1)
            investigation_name = investigation_name.strip()
            investigation_value = investigation_value.strip()

            investigation_substrings = {}

            # split only at the commas that really separate test-result pairs
            for chunk in re.split(r",\s*(?=[^,:]+?:)", investigation_value, flags=re.M):
                key, value = chunk.split(":", 1)  # only the first colon matters
                investigation_substrings[key.strip()] = value.strip()

            formatted_string += f"{investigation_name}: "
            result_keys = [
                k
                for k, v in investigation_substrings.items()
                if "result" in k.lower() and k.endswith(".")
            ]
            for result_key in result_keys:
                formatted_string += f"\n* {result_key}: {investigation_substrings[result_key]}"
                del investigation_substrings[result_key]

            if investigation_name in INVESTIGATION_SUBFIELD_SORT_ORDER:
                sorted_keys = sorted(
                    investigation_substrings.keys(),
                    key=lambda x: INVESTIGATION_SUBFIELD_SORT_ORDER[investigation_name].index(x),
                )
            else:
                sorted_keys = sorted(investigation_substrings.keys())

            for k in sorted_keys:
                formatted_string += f"\n* {k}: {investigation_substrings[k]}"
            formatted_string += "\n\n"

        return formatted_string

    @property
    def history(self) -> str:
        return format_not_recorded(HISTORY_TEMPLATE, self)

    @property
    def investigations(self) -> str:
        return format_not_recorded(INVESTIGATION_TEMPLATE, self)

    @property
    def diagnosis(self) -> str:
        return format_not_recorded(DIAGNOSIS_TEMPLATE, self)

    @property
    def treatment(self) -> str:
        return format_not_recorded(TREATMENT_TEMPLATE, self)
