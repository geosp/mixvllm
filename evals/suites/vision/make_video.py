"""Generate a 4 s, 10 fps test clip with verifiable content.

0-4 s: a red circle moves from the left edge to the right edge.
2-4 s: a blue square appears in the top-left corner.
3-4 s: the number 42 is drawn in black in the bottom-right.
"""

import cv2
import numpy as np

W, H, FPS, SECONDS = 320, 240, 10, 4
out = cv2.VideoWriter("/out/test_clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for i in range(FPS * SECONDS):
    t = i / FPS
    frame = np.full((H, W, 3), 255, np.uint8)                             # white background
    x = int(30 + (W - 60) * t / SECONDS)
    cv2.circle(frame, (x, H // 2), 22, (0, 0, 255), -1)                    # red circle (BGR)
    if t >= 2:
        cv2.rectangle(frame, (15, 15), (65, 65), (255, 0, 0), -1)          # blue square
    if t >= 3:
        cv2.putText(frame, "42", (W - 90, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    out.write(frame)
out.release()
print("frames:", FPS * SECONDS)
