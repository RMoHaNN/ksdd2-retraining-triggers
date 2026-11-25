"""Acceptance rules and decisions from recorded production batches."""

import copy
import os
import time

import torch

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.entities import Metric, Param

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


def load_records(results, training, conditions, replays, model, threshold):
    """Put the recorded metrics, decisions and initial model in a local store."""
    mlflow.set_tracking_uri("sqlite:///" + (results / "mlflow.db").as_posix())
    client = MlflowClient()
    experiments = {}
    for name in ("training", "condition-test", "replay"):
        experiment = client.get_experiment_by_name(name)
        experiments[name] = experiment.experiment_id if experiment else client.create_experiment(
            name, artifact_location=(results / "mlartifacts" / name).as_uri())
    ids = {}

    def record(experiment, name, params, values, steps=(), parent=None, document=None):
        existing = client.search_runs([experiments[experiment]],
            f"tags.`mlflow.runName` = '{name}' AND attributes.status = 'FINISHED'")
        if existing:
            return existing[0].info.run_id
        tags = {"mlflow.runName": name, "source": "paper run"}
        if parent:
            tags["mlflow.parentRunId"] = parent
        run_id = client.create_run(experiments[experiment], tags=tags).info.run_id
        now = int(time.time() * 1000)
        metrics = [Metric(key, float(value), now, 0) for key, value in values.items()]
        metrics += [Metric(key, float(value), now, step) for step, row in steps
                    for key, value in row.items()]
        try:
            client.log_batch(run_id, params=[Param(key, str(value)) for key, value in params.items()])
            for start in range(0, len(metrics), 500):
                client.log_batch(run_id, metrics=metrics[start:start + 500])
            if document:
                client.log_dict(run_id, document, "details.json")
            client.set_terminated(run_id)
        except Exception:
            client.set_terminated(run_id, "FAILED")
            raise
        return run_id

    for row in training:
        name = f"{row['regime']}-seed{row['seed']}"
        values = {"test_" + key: row[key] for key in ("ap", "recall", "precision", "f1", "auroc")}
        values["threshold"] = row["threshold"]
        steps = [(epoch["epoch"], {key: value for key, value in epoch.items() if key != "epoch"})
                 for epoch in row["history"]]
        ids[name] = record("training", name, {"regime": row["regime"], "seed": row["seed"]}, values, steps)
    for row in conditions:
        name = row["condition"]
        ids[name] = record("condition-test", name, {"condition": name},
                           {key: value for key, value in row.items() if key != "condition"})
    for name, replay_data in replays.items():
        summary = replay_data["summary"]
        steps = []
        for batch in replay_data["batches"]:
            values = {"batch_" + key: value for key, value in batch.items()
                      if isinstance(value, (int, float))}
            for feature in batch["drift_features"]:
                values.update({f"drift/{feature['feature']}/{key}": feature[key]
                               for key in ("ks_p", "psi", "flag")})
            steps.append((int(batch["batch"][2:]), values))
        ids[name] = record("replay", name, {"policy": summary["policy"], "gate": summary["gate"] or "unused"},
                           {key: value for key, value in summary.items() if key not in ("policy", "gate")},
                           steps, document=replay_data)
        for event in replay_data["events"]:
            event_name = name + "-after-" + event["after_batch"]
            record("replay", event_name,
                   {"after_batch": event["after_batch"], "condition": event["condition"],
                    "decision": event["decision"], "gate": summary["gate"]},
                   {**event["gate"], "training_seconds": event["training_seconds"]},
                   parent=ids[name], document=event["gate"])

    versions = client.search_model_versions("name = 'ksdd2-classifier'")
    if not versions:
        mlflow.set_experiment("training")
        with mlflow.start_run(run_id=ids["managed-seed0"]):
            info = mlflow.pytorch.log_model(copy.deepcopy(model).cpu(), name="v1",
                serialization_format="pickle", registered_model_name="ksdd2-classifier",
                pip_requirements=["torch==2.12.1", "torchvision==0.27.1"],
                metadata={"threshold": threshold, "source": "paper run"})
        version = info.registered_model_version
    else:
        version = versions[0].version
    client.set_registered_model_alias("ksdd2-classifier", "paper", version)
    return client, ids
