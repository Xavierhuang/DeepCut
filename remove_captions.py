import cv2
import numpy as np

cap = cv2.VideoCapture('Otodokemono.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_inpainted.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create a mask where the subtitles are.
    # For example, if we assume the subtitles are in the bottom 80px,
    # we mask that region completely:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    subtitle_height = 80
    mask[height - subtitle_height:height, 0:width] = 255
    
    # Inpaint the region
    inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    
    out.write(inpainted)

cap.release()
out.release()

