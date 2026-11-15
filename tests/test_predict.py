"""POST /predict and GET /model-info with the FLEX models in models/.

The expected values are the reference queries of the model cards written by the train repo when it exported
these models (reports/flex-outcome-0.2_model_card.md, reports/flex-extinction-0.2_model_card.md). If engine
returns something else, its copy of inference.py, its library versions or its model folders differ from train.
The API tests also start the embedding model (the app loads everything at startup).
"""
import json
import math

import pytest
from fastapi.testclient import TestClient

from app import flex
from app.main import app

ISS_AIR = {"fuel": "methanol", "pressure_atm": 1.0, "o2": 0.21, "suppressant": "none",
           "suppressant_fraction": 0.0, "droplet_d0_mm": 3.0}
MOON_WITH_CO2 = {"fuel": "methanol", "pressure_atm": 0.7, "o2": 0.34, "suppressant": "co2",
                 "suppressant_fraction": 0.2, "droplet_d0_mm": 3.5}
MOON_NO_SUPPRESSANT = {"fuel": "methanol", "pressure_atm": 0.69, "o2": 0.34, "suppressant": "none",
                       "suppressant_fraction": 0.0, "droplet_d0_mm": 3.5}
BETWEEN_PRESSURE_LEVELS = {**ISS_AIR, "pressure_atm": 0.85}
CO2_LOW_PRESSURE = {"fuel": "methanol", "pressure_atm": 0.7, "o2": 0.21, "suppressant": "co2",
                    "suppressant_fraction": 0.3, "droplet_d0_mm": 3.5}
CARD_DECIMALS = 5e-4  # the model cards print 3 decimals

# (inputs, outcome, p of that outcome, in_tested_range, out_of_range_fields, support_n, confidence, nearest cir_id)
OUTCOME_CARD = {
    "moon_like_with_co2": (MOON_WITH_CO2, "no_extinction", 0.976, False, [], 0, "low",
                           ["192F003", "192F002", "220R002", "192F001", "220R001"]),
    "iss_like_air": (ISS_AIR, "extinction", 0.587, True, [], 10, "medium",
                     ["205R001", "N01M201", "N01M101", "H01M102", "177R002"]),
    "moon_like_no_suppressant": (MOON_NO_SUPPRESSANT, "no_extinction", 0.989, True, [], 6, "high",
                                 ["192F003", "192F002", "220R002", "192F001", "220R001"]),
    "between_pressure_levels": (BETWEEN_PRESSURE_LEVELS, "extinction", 0.562, False, [], 0, "low",
                                ["N01M201", "218F001", "H01M102", "N01M101", "205R001"]),
}
# (inputs, D_ext/D0, D_ext mm, in_tested_range, out_of_range_fields, support_n, confidence, nearest cir_id)
EXTINCTION_CARD = {
    "iss_like_air": (ISS_AIR, 0.445, 1.336, True, [], 11, "high",
                     ["N01M201", "N01M101", "H01M102", "177R002", "C01M201"]),
    "co2_diluted_low_pressure": (CO2_LOW_PRESSURE, 0.626, 2.190, True, [], 6, "high",
                                 ["C10M102", "C11M201", "C09M201", "C11M401", "C08M201"]),
    "moon_like_no_suppressant": (MOON_NO_SUPPRESSANT, 0.429, 1.502, False, ["o2"], 0, "low",
                                 ["178F001", "206R001", "C08M601", "C08M201", "H10M101"]),
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


THRESHOLDS = {"outcome": 0.8300262368157266, "extinction": 1.01618202404689}  # methanol, from model_info.json


def support_of(block: dict) -> tuple:
    support = block["support"]
    return (support["in_tested_range"], support["out_of_range_fields"], support["support_n"], support["confidence"],
            [test["cir_id"] for test in support["nearest_tests"]])


@pytest.mark.parametrize("name", OUTCOME_CARD)
def test_outcome_block_matches_the_model_card(client, name):
    inputs, outcome, probability, *support = OUTCOME_CARD[name]
    response = client.post("/predict", json=inputs)
    assert response.status_code == 200, response.text
    block = response.json()["outcome"]
    assert block["prediction"]["label"] == outcome
    assert block["prediction"]["probabilities"][outcome] == pytest.approx(probability, abs=CARD_DECIMALS)
    assert sum(block["prediction"]["probabilities"].values()) == pytest.approx(1.0)
    assert support_of(block) == tuple(support)
    assert block["model"]["model_version"] == "flex-outcome-0.2"
    assert block["model"]["training_source"]["rows"] == 251


@pytest.mark.parametrize("name", EXTINCTION_CARD)
def test_extinction_block_matches_the_model_card(client, name):
    inputs, ratio, diameter_mm, *support = EXTINCTION_CARD[name]
    block = client.post("/predict", json=inputs).json()["extinction"]
    assert block["prediction"]["extinction_diameter_ratio"] == pytest.approx(ratio, abs=CARD_DECIMALS)
    assert block["prediction"]["extinction_diameter_mm"] == pytest.approx(diameter_mm, abs=CARD_DECIMALS)
    assert support_of(block) == tuple(support)
    assert block["model"]["model_version"] == "flex-extinction-0.2"
    assert block["model"]["training_source"]["rows"] == 156


def test_each_block_has_prediction_support_and_model(client):
    body = client.post("/predict", json=ISS_AIR).json()
    assert list(body) == ["outcome", "extinction"]
    for block in body.values():
        assert list(block) == ["prediction", "support", "model"]
    assert list(body["outcome"]["prediction"]) == ["label", "probabilities"]
    assert list(body["extinction"]["prediction"]) == ["extinction_diameter_ratio", "extinction_diameter_mm",
                                                      "fuel_burned_fraction"]


def test_support_carries_the_knn_threshold_of_the_queried_fuel(client):
    for name, block in client.post("/predict", json=ISS_AIR).json().items():
        support = block["support"]
        assert support["threshold"] == pytest.approx(THRESHOLDS[name])
        # in range = the closest real test is within the threshold (and every input inside its own range)
        assert support["in_tested_range"] == (support["nearest_tests"][0]["distance"] <= support["threshold"])
        within = sum(test["distance"] <= support["threshold"] for test in support["nearest_tests"])
        assert within == min(support["support_n"], len(support["nearest_tests"]))


def test_heptane_queries_use_the_heptane_threshold(client):
    support = client.post("/predict", json={**ISS_AIR, "fuel": "heptane"}).json()["outcome"]["support"]
    assert support["threshold"] == pytest.approx(0.7236060980831089)
    assert all(test["conditions"]["fuel"] == "heptane" for test in support["nearest_tests"])


def test_fuel_burned_fraction_is_one_minus_the_cubed_ratio(client):
    prediction = client.post("/predict", json=ISS_AIR).json()["extinction"]["prediction"]
    assert prediction["fuel_burned_fraction"] == pytest.approx(1 - prediction["extinction_diameter_ratio"] ** 3)
    assert 0.0 <= prediction["fuel_burned_fraction"] <= 1.0


def test_nearest_tests_carry_the_testsheet_and_the_real_conditions(client):
    for block in client.post("/predict", json=ISS_AIR).json().values():
        distances = [test["distance"] for test in block["support"]["nearest_tests"]]
        assert distances == sorted(distances)
        for test in block["support"]["nearest_tests"]:
            assert test["testsheet_id"] is None or test["testsheet_id"].startswith("FLEX-")
            assert test["test_end"] in ("extinction", "disruption", "completion")
            assert sorted(test["conditions"]) == ["co2", "d0_mm", "fuel", "he", "o2", "pressure_atm"]
            assert test["conditions"]["fuel"] == "methanol"


def test_integer_numbers_are_accepted(client):
    assert client.post("/predict", json={**ISS_AIR, "pressure_atm": 1, "droplet_d0_mm": 3}).status_code == 200


def test_value_outside_its_tested_range_is_flagged_not_rejected(client):
    response = client.post("/predict", json={**ISS_AIR, "droplet_d0_mm": 6.0})
    assert response.status_code == 200
    for block in response.json().values():
        assert block["support"]["out_of_range_fields"] == ["droplet_d0_mm"]
        assert block["support"]["in_tested_range"] is False
        assert block["support"]["confidence"] == "low"


@pytest.mark.parametrize("body", [
    {key: value for key, value in ISS_AIR.items() if key != "o2"},  # missing field
    {**ISS_AIR, "fiber": True},  # unknown field (fiber is fixed by the model, never an input)
    {**ISS_AIR, "fuel": "ethanol"},  # unknown fuel
    {**ISS_AIR, "suppressant": "argon", "suppressant_fraction": 0.2},  # unknown suppressant
    {**ISS_AIR, "o2": "0.21"},  # a number sent as a string
    {**ISS_AIR, "o2": True},  # a boolean is not a number
    {**ISS_AIR, "suppressant_fraction": 0.1},  # a fraction without a suppressant
    {**ISS_AIR, "o2": 0.6, "suppressant": "co2", "suppressant_fraction": 0.5},  # o2 + co2 > 1
])
def test_invalid_request_returns_422(client, body):
    response = client.post("/predict", json=body)
    assert response.status_code == 422, response.text


def test_invalid_inputs_explain_what_is_wrong(client):
    detail = client.post("/predict", json={**ISS_AIR, "fuel": "ethanol"}).json()["detail"]
    assert "fuel" in detail and "methanol" in detail


def test_model_info_describes_the_controls_and_both_models(client):
    body = client.get("/model-info").json()
    assert list(body["inputs"]) == list(ISS_AIR)
    # The controls cover both models: O2 up to the outcome model's 0.34, although extinction stops at 0.30.
    assert body["inputs"]["o2"]["max"] == 0.34
    assert body["extinction"]["inputs"]["o2"]["max"] == 0.30
    assert body["inputs"]["suppressant_fraction"]["by_selector"]["none"] == {"min": 0.0, "max": 0.0}
    assert body["outcome"]["model_version"] == "flex-outcome-0.2"
    assert body["extinction"]["model_version"] == "flex-extinction-0.2"
    assert body["outcome"]["outputs"] == ["label", "probabilities"]
    assert body["extinction"]["outputs"] == ["extinction_diameter_ratio", "extinction_diameter_mm",
                                             "fuel_burned_fraction"]
    assert body["outcome"]["support"]["thresholds"]["methanol"] == pytest.approx(THRESHOLDS["outcome"])
    assert body["outcome"]["metrics"]["roc_auc"] == pytest.approx(0.8926, abs=1e-4)
    assert body["extinction"]["metrics"]["mae"] == pytest.approx(0.0998, abs=1e-4)
    assert body["outcome"]["training_source"]["accession"] == "PSI-69"


def test_openapi_publishes_the_predict_contract(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"PredictRequest", "PredictResponse", "OutcomePrediction", "ExtinctionPrediction", "OutcomeValues",
            "ExtinctionValues", "Support", "ModelSource", "NearestTest", "ModelInfoResponse"} <= set(schemas)


# --- flex.py on its own (no API, no embedding model) ---

def test_merge_ranges_keeps_the_widest_range_and_every_choice():
    narrow = {"type": "float", "min": 0.12, "max": 0.30, "by_selector": {"co2": {"min": 0.1, "max": 0.4}}}
    wide = {"type": "float", "min": 0.15, "max": 0.34, "by_selector": {"co2": {"min": 0.05, "max": 0.5}}}
    assert flex.merge_ranges([narrow, wide]) == {"type": "float", "min": 0.12, "max": 0.34,
                                                 "by_selector": {"co2": {"min": 0.05, "max": 0.5}}}
    assert flex.merge_ranges([{"type": "enum", "values": ["a", "b"]},
                              {"type": "enum", "values": ["b", "c"]}])["values"] == ["a", "b", "c"]


def test_a_missing_model_folder_stops_the_service(tmp_path):
    with pytest.raises(RuntimeError, match="Missing model folder"):
        flex.load(tmp_path)


def test_another_scikit_learn_version_stops_the_service(tmp_path):
    for folder in flex.MODEL_FOLDERS.values():
        source, target = flex.DEFAULT_MODELS_DIR / folder, tmp_path / folder
        target.mkdir()
        for path in source.iterdir():
            (target / path.name).write_bytes(path.read_bytes())
    info_path = tmp_path / flex.MODEL_FOLDERS["outcome"] / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["versions"]["scikit-learn"] = "0.0.1"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scikit-learn"):
        flex.load(tmp_path)


def test_sample_inputs_are_a_valid_request():
    flex.load()
    result = flex.predict(flex.sample_inputs())
    assert set(result) == {"outcome", "extinction"}
    assert math.isfinite(result["extinction"]["prediction"]["fuel_burned_fraction"])
