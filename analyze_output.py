"""Quick analysis of output images for iteration."""
from PIL import Image
import numpy as np
import os, sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'test_output'
for f in sorted(os.listdir(out_dir)):
    if not f.endswith('.png'): continue
    img = np.array(Image.open(os.path.join(out_dir, f)).convert('RGB'))
    h, w = img.shape[:2]
    
    margin = 20
    edges = {'L': img[:, :margin, :], 'R': img[:, w-margin:, :],
             'T': img[:margin, :, :], 'B': img[h-margin:, :, :]}
    
    edge_report = []
    for k, v in edges.items():
        avg_bright = np.mean(v)
        dark_pct = np.mean(np.max(np.abs(v.astype(float) - 255), axis=2) > 40) * 100
        edge_report.append(f"{k}={avg_bright:.0f}({dark_pct:.0f}%)")
    
    gray = np.mean(img, axis=2)
    dark_mask = gray < 180
    if dark_mask.sum() > 0:
        ys, xs = np.where(dark_mask)
        cx = np.mean(xs) / w * 100
        cy = np.mean(ys) / h * 100
        # Check if product is clipped (dark pixels touching edges)
        top_clip = np.any(dark_mask[:5, :])
        bot_clip = np.any(dark_mask[-5:, :])
        left_clip = np.any(dark_mask[:, :5])
        right_clip = np.any(dark_mask[:, -5:])
        clip_str = ""
        if top_clip: clip_str += "T"
        if bot_clip: clip_str += "B"
        if left_clip: clip_str += "L"
        if right_clip: clip_str += "R"
        dark_area = dark_mask.sum() / (w*h) * 100
    else:
        cx, cy, dark_area, clip_str = 50, 50, 0, ""
    
    clip_label = f" CLIPPED={clip_str}" if clip_str else " no-clip"
    print(f"{f}: {w}x{h} | edges: {' '.join(edge_report)} | center=({cx:.0f}%,{cy:.0f}%) fill={dark_area:.0f}%{clip_label}")
