"""FLEX droplet models (PSI-69) behind POST /predict and GET /model-info.

Two model folders exported by the train repo (`python main.py train export`) and copied into models/:
- outcome: does the droplet flame go out on its own? (extinction / no_extinction)
- extinction: if it goes out, D_ext/D0 and D_ext in mm
Each folder carries its own k-NN support (how many real ISS tests back the query, and the closest ones).

The inference code is app/inference.py, a verbatim copy of train/src/flame/training/inference.py, so engine
returns exactly what the model cards in the train repo report. The models are baked into the image (no volumes).
If a folder is missing, or scikit-learn / numpy differ from the versions the models were exported with, the
service refuses to start.
"""
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from app import inference

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", DEFAULT_MODELS_DIR))
MODEL_FOLDERS = {"outcome": "flex-outcome-0.2", "extinction": "flex-extinction-0.2"}
EXPECTED_TASKS = {"outcome": "classification", "extinction": "regression"}
# The pickled objects depend on these exact versions (the others are pinned in requirements.txt too).
PINNED_LIBRARIES = ("scikit-learn", "numpy")
# Outputs engine adds on top of what the model computes.
DERIVED_OUTPUTS = {"extinction": ["fuel_burned_fraction"]}

_bundles: dict[str, dict[str, Any]] = {}


def check_versions(name: str, info: dict[str, Any]) -> None:
    for library in PINNED_LIBRARIES:
        exported, installed = info["versions"][library], version(library)
        if exported != installed:
            raise RuntimeError(
                f"{info['model_version']} was exported with {library} {exported}, but {installed} is installed. "
                f"Pin {library}=={exported} in requirements.txt (or re-export the {name} model with {installed})."
            )


def load_folder(name: str, directory: Path) -> dict[str, Any]:
    info_path = directory / inference.INFO_FILE
    if not info_path.exists():
        raise RuntimeError(f"Missing model folder {directory}. Copy train/artifacts/models/{directory.name} into "
                           f"engine/models/, or set MODELS_DIR.")
    # Check model_info.json before unpickling: a wrong scikit-learn version can fail, or silently misbehave, inside
    # joblib.load.
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["model_version"] != directory.name or info["task"] != EXPECTED_TASKS[name]:
        raise RuntimeError(f"{directory} holds {info['model_version']} ({info['task']}); expected "
                           f"{directory.name} ({EXPECTED_TASKS[name]})")
    check_versions(name, info)
    return inference.load_bundle(directory)


def load(models_dir: Path = MODELS_DIR) -> None:
    """Load both models once, at startup."""
    bundles = {name: load_folder(name, models_dir / folder) for name, folder in MODEL_FOLDERS.items()}
    names = [list(bundle["info"]["inputs"]) for bundle in bundles.values()]
    if any(inputs != names[0] for inputs in names):
        raise RuntimeError(f"The models take different inputs: {names}")
    _bundles.clear()
    _bundles.update(bundles)


def loaded() -> dict[str, dict[str, Any]]:
    if not _bundles:
        raise RuntimeError("FLEX models not loaded")
    return _bundles


def versions() -> dict[str, str]:
    return {name: bundle["info"]["model_version"] for name, bundle in loaded().items()}


def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run both models on the same inputs. Invalid inputs raise ValueError (the API turns it into a 422).

    Both blocks are always computed. Consumers show the extinction block only when outcome.outcome is
    "extinction": otherwise the droplet burned out or broke apart, and a diameter at extinction has no meaning.
    """
    bundles = loaded()
    outcome = inference.reference_predict(bundles["outcome"], inputs)
    extinction = inference.reference_predict(bundles["extinction"], inputs)
    # Share of the droplet's volume burned before the flame went out (volume scales with D^3).
    extinction["fuel_burned_fraction"] = 1.0 - extinction["extinction_diameter_ratio"] ** 3
    return {"outcome": outcome, "extinction": extinction}


def merge_ranges(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """One control per input, wide enough for every model: union of enum values, widest min/max."""
    merged = dict(specs[0])
    if merged["type"] == "enum":
        merged["values"] = list(dict.fromkeys(value for spec in specs for value in spec["values"]))
        return merged
    merged["min"] = min(spec["min"] for spec in specs)
    merged["max"] = max(spec["max"] for spec in specs)
    if "by_selector" in merged:
        merged["by_selector"] = {
            choice: {"min": min(spec["by_selector"][choice]["min"] for spec in specs),
                     "max": max(spec["by_selector"][choice]["max"] for spec in specs)}
            for choice in merged["by_selector"]
        }
    return merged


def describe(name: str, info: dict[str, Any]) -> dict[str, Any]:
    selected = info["cv"]["selected"]
    return {
        "model_version": info["model_version"],
        "task": info["task"],
        "algorithm": info["algorithm"]["estimator"],
        "inputs": info["inputs"],
        "outputs": info["outputs"] + DERIVED_OUTPUTS.get(name, []),
        "training_source": info["training_source"],
        "metrics": {metric: values["mean"] for metric, values in info["cv"]["metrics"][selected].items()},
        "metrics_unit": info["cv"]["unit"],
    }


def model_info() -> dict[str, Any]:
    infos = {name: bundle["info"] for name, bundle in loaded().items()}
    first = next(iter(infos.values()))
    return {
        # Build the controls from these: each model's own tested ranges are in its block, and a query outside
        # them is flagged by that model's out_of_range_fields / confidence, not rejected.
        "inputs": {field: merge_ranges([info["inputs"][field] for info in infos.values()])
                   for field in first["inputs"]},
        **{name: describe(name, info) for name, info in infos.items()},
    }


def sample_inputs() -> dict[str, Any]:
    """A valid request built from the input description (used by the startup self-check)."""
    info = loaded()["outcome"]["info"]
    selector = info["input_mapping"]["selector"]
    inputs = {}
    for field, spec in info["inputs"].items():
        inputs[field] = spec["values"][0] if spec["type"] == "enum" else spec["min"]
    inputs[selector["input"]] = selector["empty_value"]
    inputs[selector["value_input"]] = 0.0
    return inputs
