"""Fun, minimal, vertical pipeline diagram with hand-drawn robot doodles."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines
import numpy as np


def draw_robot(ax, cx, cy, scale=0.3, color="#888888"):
    """Draw a cute little robot at (cx, cy)."""
    s = scale
    # Head
    head = FancyBboxPatch((cx - 0.4*s, cy + 0.2*s), 0.8*s, 0.6*s,
                          boxstyle="round,pad=0.05", facecolor="white",
                          edgecolor=color, linewidth=1.0)
    ax.add_patch(head)
    # Eyes
    ax.add_patch(Circle((cx - 0.15*s, cy + 0.5*s), 0.07*s, fc=color, ec=color))
    ax.add_patch(Circle((cx + 0.15*s, cy + 0.5*s), 0.07*s, fc=color, ec=color))
    # Mouth
    ax.plot([cx - 0.12*s, cx + 0.12*s], [cy + 0.3*s, cy + 0.3*s],
            color=color, lw=1.0)
    # Antenna
    ax.plot([cx, cx], [cy + 0.8*s, cy + 1.05*s], color=color, lw=1.0)
    ax.add_patch(Circle((cx, cy + 1.08*s), 0.04*s, fc=color, ec=color))
    # Body
    body = FancyBboxPatch((cx - 0.35*s, cy - 0.5*s), 0.7*s, 0.7*s,
                          boxstyle="round,pad=0.04", facecolor="white",
                          edgecolor=color, linewidth=1.0)
    ax.add_patch(body)
    # Arms
    ax.plot([cx - 0.35*s, cx - 0.6*s], [cy + 0.0*s, cy - 0.15*s],
            color=color, lw=1.0)
    ax.plot([cx + 0.35*s, cx + 0.6*s], [cy + 0.0*s, cy - 0.15*s],
            color=color, lw=1.0)
    # Legs
    ax.plot([cx - 0.15*s, cx - 0.15*s], [cy - 0.5*s, cy - 0.72*s],
            color=color, lw=1.0)
    ax.plot([cx + 0.15*s, cx + 0.15*s], [cy - 0.5*s, cy - 0.72*s],
            color=color, lw=1.0)


fig, ax = plt.subplots(figsize=(4.5, 7.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11.5)
ax.axis("off")

# Steps
steps = [
    (10.2, "Constitution  (10 traits × 5 seed Qs)"),
    (8.9,  "Prompt expansion  (~2,500 prompts)"),
    (7.6,  "Teacher–student distillation"),
    (6.3,  "DPO pairs  (chosen / rejected)"),
    (5.0,  "LoRA DPO training  (r=64, β=0.1)"),
    (3.7,  "ΔW = BA"),
    (2.4,  "Evaluation"),
]

box_w, box_h = 6.4, 0.6
x_center = 5.0

for i, (y, label) in enumerate(steps):
    ls = "--" if i == len(steps) - 1 else "-"
    box = FancyBboxPatch(
        (x_center - box_w / 2, y - box_h / 2),
        box_w, box_h,
        boxstyle="round,pad=0.12",
        facecolor="white",
        edgecolor="#444444",
        linewidth=1.0,
        linestyle=ls,
    )
    ax.add_patch(box)
    ax.text(x_center, y, label, ha="center", va="center",
            fontsize=9, fontfamily="sans-serif", color="#222222")

# Arrows
for i in range(len(steps) - 1):
    y_from = steps[i][0] - box_h / 2 - 0.02
    y_to = steps[i + 1][0] + box_h / 2 + 0.02
    ax.annotate(
        "", xy=(x_center, y_to), xytext=(x_center, y_from),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0,
                        shrinkA=0, shrinkB=0),
    )

# Robots scattered around
draw_robot(ax, 1.0, 7.6, scale=0.35, color="#999999")   # next to distillation
draw_robot(ax, 9.0, 5.0, scale=0.30, color="#AAAAAA")   # next to training
draw_robot(ax, 1.2, 2.6, scale=0.28, color="#AAAAAA")   # next to eval

# Little speech bubble from the distillation robot
ax.annotate(
    "beep\nboop",
    xy=(1.45, 7.95),
    fontsize=6, color="#AAAAAA", fontstyle="italic",
    ha="left", va="bottom",
)

plt.tight_layout()
fig.savefig("/workspace/OpenCharacterTraining/pipeline_diagram.pdf",
            bbox_inches="tight", dpi=300)
fig.savefig("/workspace/OpenCharacterTraining/pipeline_diagram.png",
            bbox_inches="tight", dpi=300)
print("Done.")
