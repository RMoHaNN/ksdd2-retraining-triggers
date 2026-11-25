"""Acceptance rules and decisions from recorded production batches."""

from .data import CONFIG


def gate_g1(scores):
    return bool(
        scores["current_nominal_ap"] - scores["candidate_nominal_ap"]
        <= CONFIG["gates"]["G1"]["nominal_ap_loss"]
        and scores["candidate_condition_ap"] >= scores["current_condition_ap"]
    )


def gate_g2(scores):
    # G2 permits a nominal AP loss while retaining an absolute performance floor.
    return bool(
        scores["candidate_nominal_ap"] >= CONFIG["gates"]["G2"]["nominal_ap_floor"]
        and scores["candidate_condition_ap"] >= scores["current_condition_ap"]
    )


def replay(batches, events, policy, gate):
    """Apply the decision rules to recorded measurements and candidate gate checks."""
    decisions = []
    for index, batch in enumerate(batches):
        trigger = {"none": False, "drift": batch["alarm"], "quality": batch["alert"]}[policy]
        retrain = bool(trigger and index < len(batches) - 1)
        accepted = False
        if retrain:
            event = next(row for row in events if row["after_batch"] == batch["batch"])
            accepted = (gate_g1 if gate == "G1" else gate_g2)(event["gate"])
        decisions.append({"batch": batch["batch"], "retrain": retrain,
                          "accepted": accepted, "rejected": retrain and not accepted})
    return decisions
