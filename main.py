'''
Volleyball Player Tracking, ReID, Team Clustering & Tactical Mapping

Complete Pipeline:
1. Detect + track players using RT-DETR + BoT-SORT
2. Crop players per track ID
3. Extract ReID embeddings (OSNet)
4. Cluster players into teams + referee
5. Estimate camera motion (GMC)
6. Project players onto court using homography
7. Generate an annotated video with tactical map overlay
'''

import os
import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from ultralytics import RTDETR
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict
import torchreid
from torchvision import transforms

from utils import (
    project_point,
    compensate_point,
    estimate_camera_motion,
    draw_empty_court
)

VIDEO_PATH = "/kaggle/input/volleyball-match/Video2.mp4"
TRACKER_CFG = "tracker.yaml"
OUTPUT_VIDEO = "volleyball_annotated_with_map.mp4"
CROP_DIR = "player_crops"

PERSON_CLASS_ID = 0

# actual volleyball court dimensions (meters)
COURT_LENGTH = 18.0
COURT_WIDTH = 9.0

# Tactical map resolution
MAP_W = 900
MAP_H = 450
PAD = 100

SX = MAP_W / COURT_LENGTH
SY = MAP_H / COURT_WIDTH

TEAM_COLORS = {
    0: (255, 0, 0),   # Blue
    1: (0, 0, 255)    # Red
}
REF_COLOR = (255, 255, 0)


#------------- DETECTION AND TRACKING -------------

# Load RT-DETR model
model = RTDETR("rtdetr-l.pt")

# Run tracking with BoT-SORT
results = model.track(
    source=VIDEO_PATH,
    conf=0.35,
    iou=0.6,
    tracker="tracker.yaml",
    persist=True,
    verbose=False
)

#------------- PLAYER CROPPING FOR EMBEDDINGS ------------

os.makedirs(CROP_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
SAVE_EVERY = 5

for r in results:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if r.boxes is None or r.boxes.id is None:
        continue

    for box, tid, cls in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.id.cpu().numpy(),
        r.boxes.cls.cpu().numpy()
    ):
        if int(cls) != PERSON_CLASS_ID:
            continue

        if frame_idx % SAVE_EVERY != 0:
            continue

        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        pid_dir = os.path.join(CROP_DIR, f"id_{int(tid)}")
        os.makedirs(pid_dir, exist_ok=True)

        cv2.imwrite(
            os.path.join(pid_dir, f"frame_{frame_idx}.jpg"),
            crop
        )

cap.release()

#------------ ReId Embedding Extraction -------------

device = "cuda"

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
).to(device).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

track_embeddings = {}

for pid in tqdm(os.listdir(CROP_DIR)):
    pid_dir = os.path.join(CROP_DIR, pid)
    imgs = os.listdir(pid_dir)

    embeddings = []

    for img_name in imgs[:15]:
        img = cv2.imread(os.path.join(pid_dir, img_name))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = reid_model(img).cpu().numpy().flatten()
            embeddings.append(emb)

    if embeddings:
        track_embeddings[pid] = np.mean(embeddings, axis=0)

#-------------- Teams & Referee Clustering -------------

X = normalize(np.array(list(track_embeddings.values())))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=50)
labels = kmeans.fit_predict(X)

cluster_map = dict(zip(track_embeddings.keys(), labels))

#--------- Referee Identification using motion ---------

track_motion = defaultdict(list)
cap = cv2.VideoCapture(VIDEO_PATH)

for r in results:
    ret, frame = cap.read()
    if not ret:
        break

    if r.boxes is None or r.boxes.id is None:
        continue

    for box, tid, cls in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.id.cpu().numpy(),
        r.boxes.cls.cpu().numpy()
    ):
        if int(cls) != PERSON_CLASS_ID:
            continue

        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        track_motion[int(tid)].append((cx, cy))

cap.release()

motion_score = {
    tid: np.mean(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1))
    if len(pts) > 1 else 0
    for tid, pts in track_motion.items()
}

cluster_stats = defaultdict(list)
for pid, c in cluster_map.items():
    tid = int(pid.split("_")[1])
    cluster_stats[c].append(motion_score.get(tid, 0))

ref_cluster = min(
    cluster_stats,
    key=lambda c: (len(cluster_stats[c]), np.mean(cluster_stats[c]))
)

team_clusters = sorted(set(cluster_map.values()) - {ref_cluster})

team_map = {}
referee_ids = []

for pid, c in cluster_map.items():
    if c == ref_cluster:
        referee_ids.append(pid)
    else:
        team_map[pid] = team_clusters.index(c)

#----------- Homography (Manual calibration) -----------

frame_points = np.array([
    [100, 1100],
    [2850, 1125],
    [2600, 625],
    [350, 600],
    [1400, 1100],
    [1400, 625]
], dtype=np.float32)

court_points = np.array([
    [0, 0],
    [18, 0],
    [18, 9],
    [0, 9],
    [9, 0],
    [9, 9]
], dtype=np.float32)

court_points[:, 0] *= SX
court_points[:, 1] *= SY

H, _ = cv2.findHomography(frame_points, court_points)

#----------- Final Video Rendering ------------

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()
h, w = frame.shape[:2]

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

prev_gray = None
global_M = np.eye(3, dtype=np.float32)

for r in results:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        M = estimate_camera_motion(prev_gray, gray)
        Minv = cv2.invertAffineTransform(M)
        global_M = np.vstack([Minv, [0, 0, 1]]) @ global_M

    prev_gray = gray

    court_map = draw_empty_court(MAP_W, MAP_H, PAD, SX)
    frame_draw = frame.copy()

    if r.boxes is not None and r.boxes.id is not None:
        for box, tid, cls in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.id.cpu().numpy(),
            r.boxes.cls.cpu().numpy()
        ):
            if int(cls) != PERSON_CLASS_ID:
                continue

            pid = f"id_{int(tid)}"
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            cx, cy = compensate_point((cx, cy), global_M)

            H_eff = H @ global_M
            x_map, y_map = project_point((cx, cy), H_eff)
            y_map = MAP_H - y_map

            x_map = int(x_map + PAD)
            y_map = int(y_map + PAD)

            if pid in referee_ids:
                color = REF_COLOR
            else:
                color = TEAM_COLORS.get(team_map.get(pid, 0))

            cv2.circle(court_map, (x_map, y_map), 6, color, -1)

    out.write(frame_draw)

cap.release()
out.release()

print("✅ Annotated video saved:", OUTPUT_VIDEO)
