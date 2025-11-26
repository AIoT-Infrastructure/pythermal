#!/usr/bin/env python3
"""
Hybrid Adaptive+BG Mask People Counting with Kalman Tracking (Hybrid SORT-like)

- Adaptive human detection (room temp)
- Slow per-pixel background model (median init + selective slow update)
- Mask-based full-body contours (no square boxes)
- Hybrid tracker:
    - Per-track Kalman filter (x,y,vx,vy)
    - Greedy matching between Kalman predictions and detections (gating by distance)
    - Centroid smoothing (deque) for display stability
- Straight-right-door counting
- Visualization includes thermal overlay, per-person filled masks, IDs, counts, and temp info

Tuned defaults for ~120x90 thermal sensor.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from pythermal import (
    ThermalCapture,
    detect_object_centers,
    detect_humans_adaptive,
    cluster_objects,
    WIDTH,
    HEIGHT,
    TEMP_WIDTH,
    TEMP_HEIGHT,
)
from pythermal.utils import estimate_environment_temperature_v1


# -------------------------
# Background model (median init + slow selective update)
# -------------------------
class ThermalBackground:
    def __init__(self, alpha=0.02, median_init_frames=30):
        self.alpha = float(alpha)
        self.bg = None  # float32
        self.init_buf = []
        self.median_init_frames = int(median_init_frames)

    def feed_init(self, frame):
        """Return True when median init completed."""
        self.init_buf.append(frame.astype(np.float32))
        if len(self.init_buf) >= self.median_init_frames:
            stacked = np.stack(self.init_buf, axis=0)
            self.bg = np.median(stacked, axis=0).astype(np.float32)
            self.init_buf = None
            return True
        return False

    def initialized(self):
        return self.bg is not None

    def subtract(self, frame):
        """Return abs diff float32 between frame and bg (or zeros if not initialized)."""
        if not self.initialized():
            return np.zeros_like(frame, dtype=np.float32)
        return cv2.absdiff(frame.astype(np.float32), self.bg)

    def selective_update(self, frame, keep_mask=None):
        """Update background slowly except where keep_mask==255 (do not update there)."""
        if not self.initialized():
            return
        alpha = self.alpha
        if keep_mask is None:
            cv2.accumulateWeighted(frame.astype(np.float32), self.bg, alpha)
            return
        bg_updated = (1.0 - alpha) * self.bg + alpha * frame.astype(np.float32)
        fg = (keep_mask == 255)
        self.bg[~fg] = bg_updated[~fg]


# -------------------------
# Kalman-based Track
# -------------------------
class KalmanTrack:
    def __init__(self, tid, x, y, dt=1.0):
        """
        Initialize a simple constant-velocity Kalman for 2D tracking:
        state: [x, y, vx, vy]^T
        measurement: [x, y]
        """
        self.id = tid
        self.dt = float(dt)
        # Kalman filter via OpenCV
        self.kf = cv2.KalmanFilter(4, 2)
        # Transition matrix
        self.kf.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        # Process noise
        q = 1e-2
        self.kf.processNoiseCov = q * np.eye(4, dtype=np.float32)
        # Measurement noise
        r = 1e-1
        self.kf.measurementNoiseCov = r * np.eye(2, dtype=np.float32)
        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Initialize state
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)

        # smoothing centroid history for display
        self.centroids = deque(maxlen=30)
        self.centroids.append((x, y))

        self.lost_frames = 0
        self.inside = False
        self.counted_enter = False
        self.counted_exit = False

    def predict(self):
        pred = self.kf.predict()
        px, py = float(pred[0]), float(pred[1])
        return px, py

    def correct(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)
        self.centroids.append((x, y))
        self.lost_frames = 0

    def mark_lost(self):
        self.lost_frames += 1

    def smoothed(self, last_n=5):
        arr = np.array(self.centroids)
        n = min(len(arr), last_n)
        return tuple(arr[-n:].mean(axis=0))


# -------------------------
# People counter with Kalman tracks
# -------------------------
class PeopleCounterKalman:
    def __init__(self, door_width=90, max_track_distance=80, max_lost_frames=15, predict_dt=1.0):
        self.door_width = door_width
        self.max_track_distance = max_track_distance
        self.max_lost_frames = max_lost_frames
        self.predict_dt = float(predict_dt)

        self.next_id = 1
        self.tracks = []

        self.inside_count = 0
        self.total_entered = 0
        self.total_exited = 0

    def _point_in_door(self, fx, fy):
        door_left = WIDTH - self.door_width
        return fx >= door_left

    def update(self, detections):
        """
        detections: list of (fx, fy) in FRAME coords
        Algorithm:
          - predict all tracks
          - greedily match nearest detection to track prediction (gated)
          - update matched tracks with measurement (correct Kalman)
          - create new tracks for unmatched detections
          - increment lost for unmatched tracks, delete stale tracks
          - counting: based on smoothed centroid
        """
        # Predict step
        predictions = []
        for t in self.tracks:
            px, py = t.predict()
            predictions.append((t, px, py))

        assigned_dets = set()
        assigned_tracks = set()

        # Greedy matching: sort all track-det pairs by distance and assign
        pairs = []
        for ti, (t, px, py) in enumerate(predictions):
            for di, (dx, dy) in enumerate(detections):
                d = np.hypot(px - dx, py - dy)
                pairs.append((d, ti, di))
        pairs.sort(key=lambda x: x[0])

        for d, ti, di in pairs:
            t, px, py = predictions[ti]
            if t.id in assigned_tracks or di in assigned_dets:
                continue
            if d < self.max_track_distance:
                # match
                dx, dy = detections[di]
                t.correct(dx, dy)
                assigned_tracks.add(t.id)
                assigned_dets.add(di)

        # unmatched detections -> new tracks
        for i, det in enumerate(detections):
            if i in assigned_dets:
                continue
            dx, dy = det
            new_t = KalmanTrack(self.next_id, dx, dy, dt=self.predict_dt)
            new_t.inside = self._point_in_door(dx, dy)
            self.next_id += 1
            self.tracks.append(new_t)

        # For unmatched tracks, increase lost_frames
        for t in self.tracks:
            if t.id not in assigned_tracks:
                t.mark_lost()

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.lost_frames < self.max_lost_frames]

        # Counting logic
        for t in self.tracks:
            x_s, y_s = t.smoothed()
            xi, yi = int(x_s), int(y_s)
            inside_now = self._point_in_door(xi, yi)
            if (not t.inside) and inside_now:
                self.inside_count += 1
                self.total_entered += 1
                t.counted_enter = True
            elif t.inside and (not inside_now):
                self.inside_count = max(0, self.inside_count - 1)
                self.total_exited += 1
                t.counted_exit = True
            t.inside = inside_now

    def draw(self, frame):
        # door region
        overlay = frame.copy()
        door_left = WIDTH - self.door_width
        cv2.rectangle(overlay, (door_left, 0), (WIDTH - 1, HEIGHT), (0, 180, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (door_left, 0), (WIDTH - 1, HEIGHT), (0, 220, 0), 2)

        for t in self.tracks:
            x, y = t.smoothed()
            xi, yi = int(x), int(y)
            cv2.circle(frame, (xi, yi), 6, (255, 255, 255), -1)
            cv2.putText(frame, f"ID{t.id}", (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            status_color = (0, 255, 0) if t.inside else (0, 0, 255)
            cv2.circle(frame, (xi - 10, yi - 10), 7, status_color, -1)

        cv2.putText(frame, f"Inside: {self.inside_count}", (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Entered: {self.total_entered}  Exited: {self.total_exited}", (10, HEIGHT - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame


# -------------------------
# Helper functions from hybrid pipeline (kept)
# -------------------------
def temp_to_celsius(temp_array, meta_min=None, meta_max=None):
    if temp_array is None:
        return None
    if np.issubdtype(temp_array.dtype, np.floating):
        return temp_array.astype(np.float32)
    if meta_min is None or meta_max is None:
        return (temp_array.astype(np.float32) / np.max(temp_array)) * 100.0
    raw_min = float(np.min(temp_array))
    raw_max = float(np.max(temp_array))
    raw_range = raw_max - raw_min if raw_max > raw_min else 1.0
    normalized = (temp_array.astype(np.float32) - raw_min) / raw_range
    return meta_min + normalized * (meta_max - meta_min)


def build_adaptive_mask(temp_c, env_temp, min_temp_above_env, max_temp_limit):
    base_thresh = env_temp + float(min_temp_above_env)
    mask_temp = ((temp_c >= base_thresh) & (temp_c <= max_temp_limit)).astype(np.uint8) * 255
    return mask_temp


def refine_mask_and_extract_contours(adaptive_mask, bg_diff, ghost_thresh=0.8, morph_kernel_size=5, area_thresh=200):
    ghost_mask = (bg_diff >= ghost_thresh).astype(np.uint8) * 255
    combined = cv2.bitwise_and(adaptive_mask, ghost_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(combined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(combined)
    kept = []
    for c in contours:
        a = cv2.contourArea(c)
        if a >= area_thresh:
            kept.append(c)
            cv2.drawContours(final_mask, [c], -1, 255, -1)
    return final_mask, kept


def temp_to_colormap_frame(temp_c, meta_min, meta_max):
    min_t = meta_min
    max_t = meta_max
    rng = max(1e-6, max_t - min_t)
    normalized = ((temp_c - min_t) / rng) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    up = cv2.resize(normalized, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(up, cv2.COLORMAP_JET)
    return colored


def draw_person_mask_overlay(bgr_frame, contours_temp):
    mask_canvas = np.zeros_like(bgr_frame)
    for i, c in enumerate(contours_temp):
        c_scaled = c.astype(np.float32).copy()
        c_scaled[:, 0, 0] = (c_scaled[:, 0, 0] / float(TEMP_WIDTH)) * WIDTH
        c_scaled[:, 0, 1] = (c_scaled[:, 0, 1] / float(TEMP_HEIGHT)) * HEIGHT
        color = tuple(int(x) for x in np.random.randint(100, 255, size=3))
        cv2.drawContours(mask_canvas, [c_scaled.astype(np.int32)], -1, color, -1)
    blended = cv2.addWeighted(bgr_frame, 0.6, mask_canvas, 0.4, 0)
    return blended


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid Adaptive+BG Mask People Counting with Kalman Tracker")
    parser.add_argument("source", nargs="?", default=None)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--env-temp", type=float, default=None)
    parser.add_argument("--min-temp-above-env", type=float, default=2.0)
    parser.add_argument("--max-temp-limit", type=float, default=42.0)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--door-width", type=int, default=90)
    parser.add_argument("--bg-alpha", type=float, default=0.02)
    parser.add_argument("--median-init-frames", type=int, default=30)
    parser.add_argument("--ghost-thresh", type=float, default=0.8)
    parser.add_argument("--morph-k", type=int, default=5)
    parser.add_argument("--max-track-dist", type=float, default=60.0)
    parser.add_argument("--max-lost-frames", type=int, default=15)
    args = parser.parse_args()

    cap = ThermalCapture(args.source)
    is_recorded = cap.is_recorded

    bg_model = ThermalBackground(alpha=args.bg_alpha, median_init_frames=args.median_init_frames)
    counter = PeopleCounterKalman(door_width=args.door_width,
                                  max_track_distance=args.max_track_dist,
                                  max_lost_frames=args.max_lost_frames)

    window = "People Count (Hybrid Kalman)"
    frame_count = 0
    paused = False

    try:
        while True:
            if not paused:
                if not cap.has_new_frame():
                    if is_recorded:
                        time.sleep(0.01)
                        continue
                    time.sleep(0.01)
                    continue

                meta = cap.get_metadata()
                temp_raw = cap.get_temperature_array()
                yuyv = cap.get_yuyv_frame()
                if temp_raw is None or yuyv is None:
                    cap.mark_frame_read()
                    continue

                temp_c = temp_to_celsius(temp_raw, meta_min=meta.min_temp, meta_max=meta.max_temp)

                # Background median init
                if not bg_model.initialized():
                    inited = bg_model.feed_init(temp_c)
                    frame_bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
                    thermal_map = temp_to_colormap_frame(temp_c, meta.min_temp, meta.max_temp)
                    vis_init = cv2.addWeighted(frame_bgr, 0.5, thermal_map, 0.5, 0)
                    txt = f"BG init: {len(bg_model.init_buf) if bg_model.init_buf is not None else args.median_init_frames}/{args.median_init_frames}"
                    cv2.putText(vis_init, "Initializing background (median)...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(vis_init, txt, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.imshow(window, vis_init)
                    cap.mark_frame_read()
                    frame_count += 1
                    if inited:
                        print("Background initialized.")
                    continue

                # adaptive env temp
                if args.adaptive:
                    if args.env_temp is not None:
                        env_temp = args.env_temp
                    else:
                        env_temp = estimate_environment_temperature_v1(temp_c, meta.min_temp, meta.max_temp)
                else:
                    env_temp = meta.min_temp

                adaptive_mask = build_adaptive_mask(temp_c, env_temp, args.min_temp_above_env, args.max_temp_limit)
                bg_diff = bg_model.subtract(temp_c)
                final_mask, contours = refine_mask_and_extract_contours(
                    adaptive_mask, bg_diff,
                    ghost_thresh=args.ghost_thresh,
                    morph_kernel_size=args.morph_k,
                    area_thresh=args.min_area
                )

                # selective background update (do not update where final_mask==255)
                bg_model.selective_update(temp_c, keep_mask=final_mask)

                # compute centroids (frame coords)
                detections = []
                for c in contours:
                    M = cv2.moments(c)
                    if M.get("m00", 0) == 0:
                        continue
                    cx_temp = float(M["m10"] / M["m00"])
                    cy_temp = float(M["m01"] / M["m00"])
                    fx = int((cx_temp / float(TEMP_WIDTH)) * WIDTH)
                    fy = int((cy_temp / float(TEMP_HEIGHT)) * HEIGHT)
                    detections.append((fx, fy))

                # update kalman tracker & counting
                counter.update(detections)

                # visualization
                frame_bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
                thermal_map = temp_to_colormap_frame(temp_c, meta.min_temp, meta.max_temp)
                vis = cv2.addWeighted(frame_bgr, 0.5, thermal_map, 0.5, 0)
                vis = draw_person_mask_overlay(vis, contours)
                vis = counter.draw(vis)

                # info
                tmin, tmax = float(np.min(temp_c)), float(np.max(temp_c))
                info = [
                    f"Frame: {meta.seq}",
                    f"Temp: {tmin:.1f}C - {tmax:.1f}C",
                    f"Env: {env_temp:.1f}C  GhostThr: {args.ghost_thresh:.2f}C",
                    f"Masks: {len(contours)}  Tracks: {len(counter.tracks)}"
                ]
                y = 16
                for line in info:
                    cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                    y += 18

                cv2.imshow(window, vis)
                cap.mark_frame_read()
                frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done. Processed", frame_count, "frames.")


if __name__ == "__main__":
    main()
