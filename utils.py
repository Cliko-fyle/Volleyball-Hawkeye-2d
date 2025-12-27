'''
Required helper functions
'''

import cv2
import numpy as np

# function to project point on map
def project_point(pt, H):
    """
    Project a 2D point using homography matrix H
    """
    pt = np.array([[pt]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pt, H)
    return projected[0][0]

#function to compensate the camera motion
def compensate_point(pt, M3):
    """
    Apply inverse global camera motion compensation
    """
    x, y = pt
    vec = np.array([x, y, 1.0], dtype=np.float32)
    x_new, y_new, _ = M3 @ vec
    return x_new, y_new

# function to estimate the camera motion
def estimate_camera_motion(prev_gray, gray, boxes=None, classes=None):
    """
    Estimate global camera motion using optical flow,
    ignoring player regions.
    """
    mask = np.ones(prev_gray.shape, dtype=np.uint8) * 255

    if boxes is not None and classes is not None:
        for box, cls in zip(boxes, classes):
            if int(cls) == 0:  # person
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = 0

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        mask=mask,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=30
    )

    if prev_pts is None:
        return np.eye(2, 3)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None
    )

    valid_prev = prev_pts[status.flatten() == 1]
    valid_curr = curr_pts[status.flatten() == 1]

    if len(valid_prev) < 10:
        return np.eye(2, 3)

    M, _ = cv2.estimateAffinePartial2D(
        valid_prev, valid_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=3
    )

    if M is None:
        return np.eye(2, 3)

    return M

#function to creat a 2D empty court for projection
def draw_empty_court(map_w, map_h, pad, sx):
    """
    Draw a clean volleyball court tactical map
    """
    canvas = np.ones((map_h + 2 * pad, map_w + 2 * pad, 3),
                      dtype=np.uint8) * 30

    ox, oy = pad, pad

    # Court boundary
    cv2.rectangle(
        canvas,
        (ox, oy),
        (ox + map_w, oy + map_h),
        (255, 255, 255),
        2
    )

    # Net
    net_x = ox + int(9 * sx)
    cv2.line(
        canvas,
        (net_x, oy),
        (net_x, oy + map_h),
        (200, 200, 200),
        2
    )

    return canvas
