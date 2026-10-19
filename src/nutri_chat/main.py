
import argparse
import base64
import io
import json
import operator
import os
import sys
from statistics import mean
from typing import Annotated, List, Literal, TypedDict

from PIL import Image
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

# --------------------------------------------------------------------------
# Config (override with env vars)
# --------------------------------------------------------------------------
VISION_MODEL = os.getenv("VISION_MODEL", "gemma3:12b")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemma3:12b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MAX_IMAGE_SIDE = 768  # keeps vision token cost and RAM low


def vision_llm() -> ChatOllama:
    return ChatOllama(model=VISION_MODEL, base_url=OLLAMA_URL,
                      temperature=0, num_ctx=4096, keep_alive="2m")


def text_llm() -> ChatOllama:
    return ChatOllama(model=TEXT_MODEL, base_url=OLLAMA_URL,
                      temperature=0, num_ctx=4096, keep_alive="2m")


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
Category = Literal["fruit", "vegetable", "not_fruit_or_vegetable"]


class Detection(BaseModel):
    category: Category
    name: str = Field(description="Common name, e.g. 'banana'. Use 'none' if not food.")
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(description="One short sentence.")


class Nutrient(BaseModel):
    name: str
    amount_per_100g: str = Field(description="e.g. '9.0 mg' or '120 mcg'")


class Nutrition(BaseModel):
    calories_kcal_per_100g: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float
    sugar_g: float
    vitamins: List[Nutrient]
    minerals: List[Nutrient]


class Condition(BaseModel):
    condition: str
    how_it_may_help: str
    evidence: Literal["strong", "moderate", "limited"]


class Health(BaseModel):
    may_support: List[Condition]
    cautions: List[str] = Field(description="Who should limit or avoid it, allergies, interactions.")


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------
class State(TypedDict, total=False):
    image_path: str
    image_b64: str
    detections: Annotated[List[dict], operator.add]  # parallel agents append here
    verdict: dict
    nutrition: dict
    health: dict
    result: dict


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ask(llm, schema, system: str, text: str, image_b64: str | None = None):
    content = [{"type": "text", "text": text}]
    if image_b64:
        content.append({"type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}"})
    msgs = [SystemMessage(content=system), HumanMessage(content=content)]
    if schema is None:
        return llm.invoke(msgs).content
    return llm.with_structured_output(schema, method="json_schema").invoke(msgs)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def preprocess(state: State) -> dict:
    path = state["image_path"]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return {"image_b64": base64.b64encode(buf.getvalue()).decode(), "detections": []}


DIRECT_SYS = (
    "You are a produce identification expert. Classify the main item in the image. "
    "Use culinary categories: fruit (apple, tomato, banana...) or vegetable "
    "(carrot, spinach, potato, mushroom...). Anything else (cooked dishes, packaged "
    "food, toys, drawings, animals, people, objects) is 'not_fruit_or_vegetable'."
)


def agent_direct(state: State) -> dict:
    out = ask(vision_llm(), Detection, DIRECT_SYS,
              "Classify the main item in this image.", state["image_b64"])
    return {"detections": [{**out.model_dump(), "agent": "direct"}]}


def agent_describe(state: State) -> dict:
    """Two-stage: vision model only DESCRIBES, text model CLASSIFIES from words."""
    desc = ask(vision_llm(), None,
               "You describe images objectively. Do NOT name the item. Describe "
               "color, shape, size, skin/texture, whether cut/cooked/packaged, "
               "and whether it looks real, a toy, or a drawing.",
               "Describe the main object in detail.", state["image_b64"])
    out = ask(text_llm(), Detection, DIRECT_SYS,
              f"Based only on this description, classify the item:\n\n{desc}")
    return {"detections": [{**out.model_dump(), "agent": "describe_then_classify"}]}


SKEPTIC_SYS = (
    "You are a skeptical inspector. First look for reasons the main item is NOT a "
    "real, edible fruit or vegetable: toy, plastic/wax replica, painting, "
    "screen photo, cooked/processed dish, packaged product, animal, person, other "
    "object. If you find none, classify it normally as fruit or vegetable. "
    "Culinary categories apply (tomato = fruit-or-vegetable by usage; pick one)."
)


def agent_skeptic(state: State) -> dict:
    out = ask(vision_llm(), Detection, SKEPTIC_SYS,
              "Inspect this image and classify the main item.", state["image_b64"])
    return {"detections": [{**out.model_dump(), "agent": "skeptic"}]}


def consensus(state: State) -> dict:
    dets = state["detections"]
    votes: dict[str, list] = {}
    for d in dets:
        votes.setdefault(d["category"], []).append(d)
    top_cat, winners = max(votes.items(), key=lambda kv: len(kv[1]))

    if len(winners) >= 2:
        best = max(winners, key=lambda d: d["confidence"])
        category, name, method = top_cat, best["name"], "majority_vote"
        conf = mean(d["confidence"] for d in winners)
    else:  # 3-way split -> text LLM judge
        summary = json.dumps(dets, indent=2)
        j = ask(text_llm(), Detection,
                "You are a judge. Three agents disagree. Pick the most plausible "
                "final classification. Be conservative: if the evidence is weak, "
                "choose 'not_fruit_or_vegetable'.", summary)
        category, name, conf, method = j.category, j.name, j.confidence, "judge"

    is_food = category in ("fruit", "vegetable") and conf >= 0.5
    return {"verdict": {"is_fruit_or_vegetable": is_food, "category": category,
                        "name": name.lower().strip(), "confidence": round(conf, 2),
                        "method": method, "agent_votes": dets}}


def route_after_consensus(state: State) -> str:
    return "nutrition" if state["verdict"]["is_fruit_or_vegetable"] else "report"


def nutrition(state: State) -> dict:
    name = state["verdict"]["name"]
    for _ in range(2):  # retry once if values are implausible
        n = ask(text_llm(), Nutrition,
                "You are a clinical nutritionist. Give typical USDA-style values "
                "per 100 g of the RAW edible portion. List the 5-8 most significant "
                "vitamins and minerals. Be realistic; do not invent exotic values.",
                f"Nutrition profile for: {name}")
        if 0 <= n.calories_kcal_per_100g <= 900:
            break
    return {"nutrition": n.model_dump()}


def health(state: State) -> dict:
    name = state["verdict"]["name"]
    nut = json.dumps(state["nutrition"])
    h = ask(text_llm(), Health,
            "You are an evidence-aware dietitian. Using the nutrient profile, list "
            "conditions this food may help SUPPORT (never claim it cures or treats). "
            "Rate evidence honestly as strong/moderate/limited. Include cautions "
            "(e.g. potassium for kidney disease, sugar for diabetes, allergies, "
            "drug interactions).",
            f"Food: {name}\nNutrients per 100 g: {nut}")
    return {"health": h.model_dump()}


def report(state: State) -> dict:
    v = state["verdict"]
    if not v["is_fruit_or_vegetable"]:
        return {"result": {"is_fruit_or_vegetable": False,
                           "message": "This is not a fruit or vegetable.",
                           "detection": v}}
    return {"result": {
        "is_fruit_or_vegetable": True,
        "name": v["name"],
        "category": v["category"],
        "confidence": v["confidence"],
        "nutrition_per_100g": state["nutrition"],
        "health_benefits": state["health"],
        "disclaimer": "General information only, not medical advice. "
                      "Consult a doctor or dietitian for health conditions.",
        "detection": v,
    }}


# --------------------------------------------------------------------------
# Build graph
# --------------------------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("preprocess", preprocess)
    g.add_node("agent_direct", agent_direct)
    g.add_node("agent_describe", agent_describe)
    g.add_node("agent_skeptic", agent_skeptic)
    g.add_node("consensus", consensus)
    g.add_node("nutrition", nutrition)
    g.add_node("health", health)
    g.add_node("report", report)

    g.add_edge(START, "preprocess")
    for a in ("agent_direct", "agent_describe", "agent_skeptic"):
        g.add_edge("preprocess", a)           # fan-out
    g.add_edge(["agent_direct", "agent_describe", "agent_skeptic"], "consensus")  # fan-in
    g.add_conditional_edges("consensus", route_after_consensus,
                            {"nutrition": "nutrition", "report": "report"})
    g.add_edge("nutrition", "health")
    g.add_edge("health", "report")
    g.add_edge("report", END)
    return g.compile()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="path to image")
    ap.add_argument("--graph", action="store_true", help="print Mermaid diagram")
    args = ap.parse_args()

    app = build_graph()
    if args.graph:
        print(app.get_graph().draw_mermaid())
        sys.exit(0)
    if not args.image:
        ap.error("image path required")

    final = app.invoke({"image_path": args.image})
    print(json.dumps(final["result"], indent=2))