# Volleyball-Hawkeye-2d

**Volleyball Player Tracking, Re-Identification, Team Clustering, and Tactical Court Mapping using RT-DETR, BoT-SORT, and Homography**

**OVERVIEW**

This project performs end-to-end volleyball analytics from broadcast video:

- Player detection & tracking
- Team clustering
- Referee separation
- Camera motion compensation
- Tactical court projection


**APPROACH**

- **Detection & Tracking**: RT-DETR + BoT-SORT
- **ReID**: OSNet embeddings
- **Clustering**: KMeans (2 teams + referee)
- **Referee Identification**: Smallest cluster + lowest motion
- **Court Mapping**: Manual homography (short video)
- **Camera Motion Compensation**: Optical flow–based GMC
- **Visualization**: Real-time tactical map overlay


**ASSUMPTIONS**

- Static court geometry
- Single broadcast camera view
- Manual homography points


**LIMITATIONS**

- Weak "Ball" detection due to motion blur
- Manual court calibration required
- Broadcast zoom affects the homography
- Camera view affects mapping (depth-based view vs side view)
- Not real-time (offline processing)

**OUTPUT**

- Annotated video with:
  - Player IDs
  - Team and Referee colour segmentation
  - Tactical 2D court overlay
 

**SAMPLE OUPUTS**

<img width="1919" height="953" alt="Screenshot 2025-12-27 164401" src="https://github.com/user-attachments/assets/a3a3be5d-7911-4494-a3e9-55857018c9e8" />
<img width="1919" height="899" alt="Screenshot 2025-12-27 164446" src="https://github.com/user-attachments/assets/4007c749-175f-4589-89bc-14241f1d7360" />
<img width="1919" height="895" alt="Screenshot 2025-12-27 164315" src="https://github.com/user-attachments/assets/9a748695-5006-45ff-af79-621b601b2fd2" />

