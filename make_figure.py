"""Render the DTMIR-Pro+ method diagram to a PNG.

Usage:  python make_figure.py
Output: dtmir_pro_method_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W, H = 16, 9
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

def box(x, y, w, h, text, fc="#f0f4ff", ec="#2b4cb6", lw=1.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)
    return (x + w/2, y + h/2)

def tag(x, y, text, fc="#ffffff", ec="#555555"):
    p = FancyBboxPatch((x, y), 1.9, 0.5, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.1)
    ax.add_patch(p)
    ax.text(x + 0.95, y + 0.25, text, ha="center", va="center", fontsize=8.5)

def arrow(x1, y1, x2, y2, color="#444"):
    a = FancyArrowPatch((x1, y1), (x2, y2), color=color, arrowstyle="->",
                        lw=1.7, mutation_aspect=4)
    ax.add_patch(a)

def vline(x, ytop, ybot, color="#444"):
    a = FancyArrowPatch((x, ytop), (x, ybot), color=color, arrowstyle="-",
                        lw=1.7, mutation_aspect=4)
    ax.add_patch(a)

# --- Input ---
box(0.4, 6.4, 2.0, 1.3, "Degraded\nInput $I_x$\n(Haze / Rain / Snow)")
in_ctr = 1.4, 7.2

# --- Translation stage (3 streams) ---
tx = 3.2
box(tx-1.1, 6.5, 2.2, 0.95, "Weather-Coded\nPrompt + DynConv", fc="#fff3e0", ec="#e65100")
tag(tx-1.1, 6.0, "Haze", fc="#fff3e0", ec="#e65100")
box(tx-1.1, 4.5, 2.2, 0.95, "Weather-Coded\nPrompt + DynConv", fc="#e8f0fe", ec="#1565c0")
tag(tx-1.1, 4.0, "Rain", fc="#e8f0fe", ec="#1565c0")
box(tx-1.1, 2.5, 2.2, 0.95, "Weather-Coded\nPrompt + DynConv", fc="#fce8e6", ec="#b71c1c")
tag(tx-1.1, 2.0, "Snow", fc="#fce8e6", ec="#b71c1c")
# stacked stream outputs
out_y = 0.9
out_ctr = box(tx, out_y, 2.0, 1.1, "Translated streams\n$T_H, T_R, T_S$", fc="#ffffff", ec="#444")
vline(tx+1.0, 6.9, out_y+1.2, "#e65100")
vline(tx+1.0, 4.9, out_y+1.2, "#1565c0")
vline(tx+1.0, 2.9, out_y+1.2, "#b71c1c")
vline(tx+1.0, out_y+1.2, out_y+0.1, "#444")

# --- SBFR routing ---
rx = 9.8
box(rx-0.8, out_y+0.9, 1.6, 1.1, "SBFR Routing\n(last enc. layer)", fc="#e8f5e9", ec="#2e7d32")
tag(rx-0.8, out_y+2.1, "Hard: argmin SSIM")
tag(rx-0.8, out_y+1.5, "Soft: learned")
arrow(out_ctr[0]+1.0, out_y+1.45, rx-0.8, out_y+1.2)   # translations feed routing

# route feature into restorer
rout = (rx+0.8, out_y+1.45)

# translations also go to restorer
arrow(out_ctr[0]+1.0, out_y+0.1, 13.2, 5.6)

# --- Restoration ---
rest_x = 12.8
box(rest_x-1.6, 6.4, 3.2, 1.0, "Extract $F_H, F_R, F_S$", fc="#f3e5f5", ec="#6a1b9a")
arrow(rout[0]+0.2, rout[1], rest_x-0.4, 6.4+0.5)
box(rest_x-1.6, 5.2, 3.2, 1.0, "MWF pairs $\\to F_{HR}, F_{RS}, F_{HS}$", fc="#f3e5f5", ec="#6a1b9a")
box(rest_x-1.6, 4.0, 3.2, 1.0, "3 parallel encoder branches $\\to E^i_{HR/RS/HS}$", fc="#f3e5f5", ec="#6a1b9a")
box(rest_x-1.6, 2.9, 3.2, 1.0, "Concat per level with $\\uparrow E_{DT}$", fc="#e0f2f1", ec="#00695c")
box(rest_x-1.6, 1.2, 3.2, 1.3, "Restored\nClean Image $\\hat{Y}$", fc="#fff9c4", ec="#ff8f00")
arrow(rest_x, 6.4+0.4, rest_x-0.4, 5.4)
arrow(rest_x, 5.2+0.4, rest_x-0.4, 4.2)
arrow(rest_x, 4.0+0.4, rest_x-0.4, 3.2)
arrow(rest_x, 2.9+0.4, rest_x-0.4, 2.1)

# --- Bottom: losses + metrics ---
by = -0.3
box(0.3, by, 2.8, 0.8, "$L_{adv}$\nadversarial", fc="#fff8e1", ec="#ff8f00")
box(3.3, by, 2.8, 0.8, "$L_{cont}$\ncontrastive", fc="#fff8e1", ec="#ff8f00")
box(6.3, by, 3.0, 0.8, "$L_{rec}$\n$L_1$+perceptual", fc="#fff8e1", ec="#ff8f00")
box(9.5, by, 3.2, 0.8, "Metrics: PSNR / SSIM / LPIPS / FID", fc="#ede7f6", ec="#512da8")

fig.text(0.5, 0.03,
         "Fig. 1. DTMIR-Pro+ overview. A degraded input is translated into three weather-specific "
         "streams using Weather-Coded Prompts (WCP). SBFR routes the last translation-encoder layer "
         "(hard via argmin SSIM, or soft via a learned 3-way blend). The restorer extracts per-stream "
         "features, fuses each pair with the Multi-Weather Fusion (MWF) block, encodes them through "
         "three parallel branches, and at each decoder level concatenates the MWF features with the "
         "upsampled routed feature to restore a clean image. Training jointly optimizes adversarial, "
         "contrastive, and L1/perceptual reconstruction losses; we evaluate with PSNR, SSIM, LPIPS, "
         "and FID.",
         ha="center", va="bottom", fontsize=7.4, wrap=True)

plt.savefig("dtmir_pro_method_diagram.png", dpi=180, bbox_inches="tight",
            facecolor="white")
print("saved dtmir_pro_method_diagram.png")
