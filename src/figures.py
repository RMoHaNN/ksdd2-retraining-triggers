"""Paper figures and the classifier architecture, drawn with matplotlib."""

import copy

import torch
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

from .data import CONFIG, IMG_H, IMG_W, KSDD2, ROOT
from .experiments import gate_g1, gate_g2

COLORS = {"none": "#8c8c8c", "G1": "#d55e00", "G2": "#0072b2"}
OUTPUT = ROOT / "figures"
plt.switch_backend("Agg")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
    }
)


def save(figure, name):
    OUTPUT.mkdir(exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = OUTPUT / f"{name}.{extension}"
        figure.savefig(path, dpi=CONFIG["figures"]["dpi"], bbox_inches="tight", pad_inches=0.03)
        paths.append(path)
    plt.close(figure)
    return paths


def box(axis, center, text, width=0.27, height=0.075, fontsize=8):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.005,rounding_size=0.005",
        linewidth=0.6,
        edgecolor="black",
        facecolor="white",
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def arrow(axis, start, end):
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=7, linewidth=0.6, color="black"
        )
    )


def fig_workflow():
    figure, axis = plt.subplots(figsize=(4.3, 5.1))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    for center, text in [
        ((0.25, 0.93), "Batch"),
        ((0.75, 0.93), "Current model"),
        ((0.25, 0.78), "Drift monitor\nwithout labels"),
        ((0.75, 0.78), "Quality monitor\nwith labels"),
        ((0.5, 0.64), "Trigger"),
        ((0.5, 0.52), "Retrained candidate"),
        ((0.5, 0.40), "Acceptance gate"),
        ((0.5, 0.28), "Replace or keep"),
    ]:
        box(axis, center, text)
    arrow(axis, (0.39, 0.93), (0.61, 0.93))
    arrow(axis, (0.75, 0.89), (0.75, 0.82))
    axis.plot([0.75, 0.75, 0.25], [0.89, 0.86, 0.86], "k-", lw=0.6)
    arrow(axis, (0.25, 0.86), (0.25, 0.82))
    for x, end in [(0.25, 0.36), (0.75, 0.64)]:
        axis.plot([x, x], [0.74, 0.64], "k-", lw=0.6)
        arrow(axis, (x, 0.64), (end, 0.64))
    for top, bottom in [(0.60, 0.56), (0.48, 0.44), (0.36, 0.32)]:
        arrow(axis, (0.5, top), (0.5, bottom))
    axis.text(0.52, 0.58, "triggered", fontsize=6.5, va="center")
    axis.plot([0.64, 0.96, 0.96], [0.64, 0.64, 0.28], "k-", lw=0.6)
    arrow(axis, (0.96, 0.28), (0.64, 0.28))
    axis.text(
        0.94,
        0.46,
        "not triggered: keep",
        rotation=90,
        fontsize=6.5,
        ha="right",
        va="center",
        backgroundcolor="white",
    )
    axis.plot([0.36, 0.04, 0.04], [0.28, 0.28, 0.93], "k-", lw=0.6)
    arrow(axis, (0.04, 0.93), (0.11, 0.93))
    axis.text(0.06, 0.49, "next batch", rotation=90, fontsize=6.5, va="center")
    table = axis.table(
        cellText=[
            ["No retraining", "Shared baseline", "Gate unused"],
            ["Data drift", "Drift + G1", "Drift + G2"],
            ["Quality alert", "Quality + G1", "Quality + G2"],
        ],
        colLabels=["Trigger", "G1", "G2"],
        cellLoc="left",
        bbox=[0.06, 0, 0.9, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for cell in table.get_celld().values():
        cell.visible_edges = "horizontal"
        cell.set_linewidth(0.4)
    return save(figure, "fig_workflow")


def fig_architecture(model):
    network = copy.deepcopy(model).cpu().eval()
    shapes, hooks = {}, []
    for name in ("conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"):
        hooks.append(
            getattr(network, name).register_forward_hook(
                lambda module, inputs, output, key=name: shapes.update(
                    {key: list(output.shape[1:])}
                )
            )
        )
    try:
        with torch.no_grad():
            network(torch.zeros(1, 3, IMG_H, IMG_W))
    finally:
        for hook in hooks:
            hook.remove()
    figure, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.set(xlim=(-0.02, 1.02), ylim=(0, 1))
    axis.axis("off")
    axis.text(0.5, 0.97, "ResNet-18 backbone (ImageNet pretrained)", ha="center", fontsize=10)
    names = ["Input", "conv1", "layer1", "layer2", "layer3", "layer4"]
    positions = np.linspace(0.08, 0.92, len(names))
    for index, (x, name) in enumerate(zip(positions, names)):
        shape = [3, IMG_H, IMG_W] if name == "Input" else shapes[name]
        label = f"{name}\n{shape[1]} x {shape[2]} x {shape[0]}"
        box(axis, (x, 0.81), label, width=0.135, height=0.13)
        if index:
            arrow(axis, (positions[index - 1] + 0.07, 0.81), (x - 0.07, 0.81))
    axis.plot([0.92, 0.92, 0.08], [0.74, 0.68, 0.68], "k-", lw=0.6)
    arrow(axis, (0.08, 0.68), (0.08, 0.60))
    labels = [
        f"Global average\npooling ({shapes['avgpool'][0]})",
        f"Linear layer\n{shapes['fc'][0]} logits",
        "Softmax\ndefect score",
        "Validation F1\nthreshold",
        "Defective /\nnon-defective",
    ]
    for index, (x, label) in enumerate(zip(np.linspace(0.08, 0.92, 5), labels)):
        box(axis, (x, 0.53), label, width=0.155, height=0.13)
        if index:
            arrow(axis, (0.08 + (index - 1) * 0.21 + 0.08, 0.53), (x - 0.08, 0.53))
    box(
        axis,
        (0.25, 0.27),
        f"{shapes['avgpool'][0]}-dim pooled features\nPCA ({CONFIG['monitor']['pca_components']})",
        width=0.25,
        height=0.12,
    )
    axis.plot([0.08, 0.08], [0.46, 0.27], "k-", lw=0.6)
    arrow(axis, (0.08, 0.27), (0.12, 0.27))
    box(axis, (0.25, 0.08), "Image brightness, contrast, sharpness", width=0.35, height=0.09)
    box(axis, (0.68, 0.20), "Data drift monitor\nKS + PSI on 13 features", width=0.30, height=0.15)
    arrow(axis, (0.38, 0.27), (0.53, 0.23))
    arrow(axis, (0.43, 0.08), (0.53, 0.16))
    return save(figure, "fig_architecture"), shapes


def fig_conditions(model, threshold, test):
    item = next(
        item for item in test if item[0].endswith(f"{CONFIG['figures']['condition_image']}.png")
    )
    conditions = ["nominal", "lighting", "lighting_noise", "defocus"]
    datasets = [KSDD2([item], condition=condition) for condition in conditions]
    with torch.inference_mode():
        scores = torch.softmax(
            copy.deepcopy(model).cpu().eval()(torch.stack([dataset[0][0] for dataset in datasets])),
            1,
        )[:, 1].tolist()
    mask = np.asarray(Image.open(item[1])) > 0
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    yy, xx = np.where(mask)
    x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)
    height, width = mask.shape
    side = min(
        width,
        height,
        max(
            CONFIG["figures"]["crop_min_side"],
            int(np.ceil(max(x1 - x0, y1 - y0) * CONFIG["figures"]["crop_scale"])),
        ),
    )
    left = min(max(0, round((x0 + x1 - side) / 2)), width - side)
    top = min(max(0, round((y0 + y1 - side) / 2)), height - side)
    figure, axes = plt.subplots(1, 4, figsize=(7.05, 2.4))
    for axis, dataset, condition, score in zip(axes, datasets, conditions, scores):
        image, _ = dataset.load(0)
        axis.imshow(
            np.asarray(image)[top : top + side, left : left + side], interpolation="nearest"
        )
        axis.contour(
            mask[top : top + side, left : left + side].astype(float),
            levels=[0.5],
            colors=["#f2f2f2"],
            linewidths=0.35,
        )
        axis.set_title(CONFIG["conditions"][condition]["label"], fontsize=10, pad=5)
        detected = "detected" if score >= threshold else "missed"
        axis.set_xlabel(f"Defect score {score:.3f}\n{detected}", fontsize=9.5, labelpad=5)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.28, top=0.86, wspace=0.13)
    figure.text(
        0.5,
        0.01,
        f"v1 threshold: {threshold:.4f}. Scores use full images; crops are for display.",
        ha="center",
        fontsize=8,
    )
    record = {
        "image_id": CONFIG["figures"]["condition_image"],
        "threshold": threshold,
        "scores": dict(zip(conditions, scores)),
        "crop": [left, top, left + side, top + side],
    }
    return save(figure, "fig_conditions"), record


def fig_replay(replays):
    figure, (axis, drift_axis) = plt.subplots(
        2,
        1,
        figsize=(3.45, 3.05),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        layout="constrained",
    )
    bands = [
        (0.5, 2.5, "Nominal"),
        (2.5, 4.5, "Lighting"),
        (4.5, 5.5, "+ noise"),
        (5.5, 8.5, "Defocus"),
    ]
    for panel in (axis, drift_axis):
        for index, (left, right, _) in enumerate(bands):
            if index % 2:
                panel.axvspan(left, right, color="0.93", lw=0, zorder=0)
    for key, label, color, marker, size in [
        ("none", "No retraining", "none", "s", 3.0),
        ("quality-G1", "Quality, gate G1", "G1", "D", 2.7),
        ("quality-G2", "Quality, gate G2", "G2", "o", 3.0),
    ]:
        rows = replays[key]["batches"]
        axis.plot(
            [int(row["batch"][2:]) for row in rows],
            [row["recall"] for row in rows],
            label=label,
            color=COLORS[color],
            marker=marker,
            markersize=size,
            lw=1.1,
        )
    for gate, y in [("G1", 0.06), ("G2", 0.14)]:
        for event in replays[f"quality-{gate}"]["events"]:
            accepted = event["decision"] == "accepted"
            axis.plot(
                int(event["after_batch"][2:]) + 0.5,
                y,
                marker="^" if accepted else "x",
                markersize=4.2,
                mew=1.1,
                color=COLORS[gate],
                ls="none",
                zorder=4,
            )
        axis.text(4.85, y, gate, fontsize=6.5, ha="right", va="center", color=COLORS[gate])
    axis.plot([], [], marker="^", color="0.2", ls="none", markersize=4, label="Candidate promoted")
    axis.plot(
        [],
        [],
        marker="x",
        color="0.2",
        ls="none",
        markersize=4,
        mew=1.1,
        label="Candidate rejected",
    )
    axis.set(ylim=(0, 1.05), ylabel="Defect recall (%)")
    axis.yaxis.set_major_formatter(PercentFormatter(1, symbol=""))
    axis.legend(loc="center left", bbox_to_anchor=(0, 0.5), frameon=False, handlelength=1.8)
    for left, right, label in bands:
        axis.text(
            (left + right) / 2,
            1.03,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            transform=axis.get_xaxis_transform(),
        )
    rows = replays["none"]["batches"]
    drift_axis.bar(
        [int(row["batch"][2:]) for row in rows],
        [row["flagged"] for row in rows],
        color="#7f7f7f",
        width=0.55,
    )
    limit = CONFIG["monitor"]["alarm_features"]
    drift_axis.axhline(limit, color="k", lw=0.6, ls=(0, (4, 2)))
    drift_axis.text(0.6, limit + 0.6, "Alarm threshold", fontsize=7)
    drift_axis.set(
        ylabel="Flagged\nfeatures",
        ylim=(0, 14),
        yticks=[0, 4, 8, 12],
        xlabel="Production batch",
        xlim=(0.5, 8.5),
        xticks=range(1, 9),
        xticklabels=[f"PB{i}" for i in range(1, 9)],
    )
    for panel in (axis, drift_axis):
        panel.spines[["top", "right"]].set_visible(False)
    return save(figure, "fig_replay")


def fig_gate_check(replays):
    batch = CONFIG["figures"]["gate_batch"]
    event = next(
        (event for event in replays["quality-G1"]["events"] if event["after_batch"] == f"PB{batch}"), None
    )
    figure, axis = plt.subplots(figsize=(4.0, 4.8))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    if event is None:
        axis.text(0.5, 0.5, f"No quality G1 candidate after PB{batch} in this run.", ha="center")
        return save(figure, "fig_gate_check"), {"available": False}
    check = event["gate"]
    box(axis, (0.5, 0.94), f"Same candidate after PB{batch} (G1 replay)", width=0.94)
    nominal = (
        f"AP {100 * check['current_nominal_ap']:.1f}% to {100 * check['candidate_nominal_ap']:.1f}%"
    )
    changed = (
        f"AP {100 * check['current_condition_ap']:.1f}% to "
        f"{100 * check['candidate_condition_ap']:.1f}%"
    )
    loss = 100 * (check["current_nominal_ap"] - check["candidate_nominal_ap"])
    limit = 100 * CONFIG["gates"]["G1"]["nominal_ap_loss"]
    floor = 100 * CONFIG["gates"]["G2"]["nominal_ap_floor"]
    checks = {"G1": gate_g1(check), "G2": gate_g2(check)}
    for x, gate in [(0.25, "G1"), (0.75, "G2")]:
        original_ok = (
            loss <= limit if gate == "G1" else 100 * check["candidate_nominal_ap"] >= floor
        )
        condition_ok = check["candidate_condition_ap"] >= check["current_condition_ap"]
        axis.text(
            x,
            0.83,
            f"{gate}: " + ("recorded decision" if gate == "G1" else "rule check"),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
        rule = (
            f"Loss {loss:.1f} points; limit {limit:g}"
            if gate == "G1"
            else f"Nominal AP floor {floor:g}%"
        )
        box(
            axis,
            (x, 0.67),
            "A: nominal images\n"
            + nominal
            + "\n"
            + rule
            + "\n"
            + ("Pass" if original_ok else "Fail"),
            width=0.46,
            height=0.19,
        )
        box(
            axis,
            (x, 0.45),
            "B: changed-condition images\n" + changed + "\n" + ("Pass" if condition_ok else "Fail"),
            width=0.46,
            height=0.14,
        )
        decision = "Accepted" if checks[gate] else "Rejected"
        if gate == "G2":
            decision = "Would be accepted" if checks[gate] else "Would be rejected"
        box(axis, (x, 0.29), decision, width=0.46)
        arrow(axis, (0.5, 0.90), (x, 0.86))
        for top, bottom in [(0.80, 0.77), (0.57, 0.53), (0.37, 0.33)]:
            arrow(axis, (x, top), (x, bottom))
    axis.plot([0.01, 0.99], [0.21, 0.21], color="black", lw=0.4, ls="--")
    axis.text(0.5, 0.17, "Recorded recall from separate replays", ha="center", fontsize=8)
    for x, gate in [(0.25, "G1"), (0.75, "G2")]:
        rows = replays[f"quality-{gate}"]["batches"]
        text = ";\n".join(f"{row['batch']} {100 * row['recall']:.1f}%" for row in rows[batch:])
        axis.text(x, 0.07, f"{gate}: {text}", ha="center", va="center", fontsize=9)
    return save(figure, "fig_gate_check"), {"available": True, "candidate": check, "checks": checks}


def make_all(model, threshold, test, replays):
    paths = {"fig_workflow": fig_workflow()}
    paths["fig_architecture"], shapes = fig_architecture(model)
    condition_scores = None
    if test:
        paths["fig_conditions"], condition_scores = fig_conditions(model, threshold, test)
    paths["fig_replay"] = fig_replay(replays)
    paths["fig_gate_check"], gate_check = fig_gate_check(replays)
    return paths, {
        "layer_shapes": shapes,
        "condition_scores": condition_scores,
        "gate_check": gate_check,
    }
