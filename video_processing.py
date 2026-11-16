import gc
import cv2
import io
import os
import queue
import tempfile
import textwrap
import threading
import traceback
import uuid
from pathlib import Path

import torch
from config import meets_pole_confidence, is_pole_label

from db import get_uploaded_video

# In-memory dictionary tracking task status
# Format: { task_id: { "status": "queued"|"processing"|"done"|"failed", "progress": int, "results": list, "error": str } }
tasks = {}
task_lock = threading.Lock()
video_queue = queue.Queue()


TRACKER_CONFIG_YAML = textwrap.dedent("""
tracker_type: bytetrack
track_high_thresh: 0.20
track_low_thresh: 0.05
new_track_thresh: 0.10
track_buffer: 90
match_thresh: 0.8
fuse_score: True
""").strip() + "\n"
_TRACKER_CONFIG_PATH = None


def _ensure_tracker_config():
    global _TRACKER_CONFIG_PATH
    if _TRACKER_CONFIG_PATH and os.path.exists(_TRACKER_CONFIG_PATH):
        return _TRACKER_CONFIG_PATH

    tracker_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", prefix="bytetrack_")
    tracker_file.write(TRACKER_CONFIG_YAML.encode("utf-8"))
    tracker_file.close()
    _TRACKER_CONFIG_PATH = tracker_file.name
    return _TRACKER_CONFIG_PATH


def _match_keyword(cls_name: str, category: str) -> bool:
    from config import OBB_CLASS_KEYWORDS

    name_lower = cls_name.lower()
    return any(kw.lower() in name_lower for kw in OBB_CLASS_KEYWORDS.get(category, []))


def get_pole_detections(results, names):
    poles = []
    if not results:
        return poles

    result = results[0]
    if hasattr(result, 'boxes') and result.boxes is not None:
        for box in result.boxes:
            if box.cls is None or len(box.cls) == 0:
                continue
            cls_idx = int(box.cls[0])
            cls_name = names[cls_idx]
            if _match_keyword(cls_name, "pole"):
                track_id = int(box.id[0]) if (hasattr(box, 'id') and box.id is not None) else None
                conf = float(box.conf[0])
                if not meets_pole_confidence(conf):
                    continue
                bbox = box.xyxy[0].cpu().numpy().tolist()
                poles.append({
                    "track_id": track_id,
                    "conf": conf,
                    "bbox": bbox,
                    "class_name": cls_name,
                })
    return poles


def calculate_sharpness(frame, bbox):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if (x2 - x1) > 10 and (y2 - y1) > 10:
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return 0.0


def reset_yolo_tracker(model):
    if hasattr(model, 'predictor') and model.predictor:
        if hasattr(model.predictor, 'trackers') and model.predictor.trackers:
            for tracker in model.predictor.trackers:
                if hasattr(tracker, 'reset'):
                    tracker.reset()


def extract_best_pole_frames(video_path, pipeline_engine, target_fps=2, progress_callback=None):
    if hasattr(pipeline_engine, 'names'):
        model = pipeline_engine
    else:
        model = pipeline_engine.component_model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    frame_step = max(1, int(fps / target_fps))
    reset_yolo_tracker(model)

    track_metadata = {}
    frame_idx = 0
    img_w, img_h = 1920, 1080
    tracker_path = _ensure_tracker_config()

    print(f"[VideoProcessor] Starting Pass 1 tracking on {video_path} (Total frames: {total_frames}, Step: {frame_step})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        img_h, img_w = frame.shape[:2]
        if progress_callback and total_frames > 0:
            progress_callback(min(1.0, frame_idx / total_frames))

        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker_path,
            conf=0.05,
            iou=0.45,
            verbose=False,
            imgsz=640,
            classes=[0, 1],
        )

        poles = get_pole_detections(results, model.names)
        for pole in poles:
            tid = pole["track_id"]
            if tid is None:
                continue

            px1, py1, px2, py2 = pole["bbox"]
            pole_height = py2 - py1
            pole_center_x = (px1 + px2) / 2.0
            centrality = 1.0 - abs(pole_center_x - img_w / 2.0) / (img_w / 2.0)
            rel_height = pole_height / img_h
            sharpness = calculate_sharpness(frame, pole["bbox"])

            track_metadata.setdefault(tid, []).append({
                "frame_idx": frame_idx,
                "timestamp_sec": float(frame_idx) / fps,
                "bbox": pole["bbox"],
                "conf": pole["conf"],
                "centrality": centrality,
                "rel_height": rel_height,
                "sharpness": sharpness,
            })

        frame_idx += 1
    cap.release()

    filtered_tracks = {tid: entries for tid, entries in track_metadata.items() if len(entries) >= 2}
    print(f"[VideoProcessor] Tracking Pass 1 complete. Detected {len(track_metadata)} raw tracks. {len(filtered_tracks)} survived short-lived filter.")
    if not filtered_tracks:
        return {}

    tids = sorted(list(filtered_tracks.keys()))
    merged_map = {tid: tid for tid in tids}

    for index, tid1 in enumerate(tids):
        entries1 = filtered_tracks[tid1]
        frames1 = [entry["frame_idx"] for entry in entries1]
        _, end1 = min(frames1), max(frames1)

        for tid2 in tids[index + 1:]:
            entries2 = filtered_tracks[tid2]
            frames2 = [entry["frame_idx"] for entry in entries2]
            start2, _ = min(frames2), max(frames2)
            overlapping_frames = set(frames1).intersection(set(frames2))

            if overlapping_frames:
                ious = []
                for frame_no in overlapping_frames:
                    box1 = next(entry["bbox"] for entry in entries1 if entry["frame_idx"] == frame_no)
                    box2 = next(entry["bbox"] for entry in entries2 if entry["frame_idx"] == frame_no)
                    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
                    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    union = area1 + area2 - inter
                    ious.append(inter / union if union > 0 else 0)
                if (sum(ious) / len(ious)) > 0.45:
                    merged_map[tid2] = merged_map[tid1]
            else:
                earlier = entries1 if end1 <= start2 else entries2
                later = entries2 if end1 <= start2 else entries1
                box_last1 = earlier[-1]["bbox"]
                box_first2 = later[0]["bbox"]
                x_last1 = (box_last1[0] + box_last1[2]) / 2.0
                x_first2 = (box_first2[0] + box_first2[2]) / 2.0
                t_last1 = earlier[-1]["timestamp_sec"]
                t_first2 = later[0]["timestamp_sec"]
                gap_sec = t_first2 - t_last1

                if gap_sec < 1.5:
                    vx_sec = 0.0
                    if len(earlier) >= 2:
                        x_first1 = (earlier[0]["bbox"][0] + earlier[0]["bbox"][2]) / 2.0
                        t_first1 = earlier[0]["timestamp_sec"]
                        dt = t_last1 - t_first1
                        if dt > 0:
                            vx_sec = (x_last1 - x_first1) / dt
                    max_vx_sec = 1050.0
                    if abs(vx_sec) > max_vx_sec:
                        vx_sec = max_vx_sec if vx_sec > 0 else -max_vx_sec
                    projected_x = x_last1 + vx_sec * gap_sec
                    diff = abs(x_first2 - projected_x)
                    direct_dist = abs(x_first2 - x_last1)
                    if diff < (img_w * 0.18) or direct_dist < (img_w * 0.15):
                        y_last1 = (box_last1[1] + box_last1[3]) / 2.0
                        y_first2 = (box_first2[1] + box_first2[3]) / 2.0
                        if abs(y_first2 - y_last1) < (img_h * 0.20):
                            merged_map[tid2] = merged_map[tid1]

    merged_tracks = {}
    for tid, entries in filtered_tracks.items():
        resolved_tid = merged_map[tid]
        while resolved_tid != merged_map[resolved_tid]:
            resolved_tid = merged_map[resolved_tid]
        merged_tracks.setdefault(resolved_tid, []).extend(entries)

    for tid in merged_tracks:
        merged_tracks[tid].sort(key=lambda entry: entry["frame_idx"])

    print(f"[VideoProcessor] Track merging completed: reduced {len(filtered_tracks)} tracks to {len(merged_tracks)} unique poles.")

    best_frame_indices = {}
    best_metadata = {}
    for tid, entries in merged_tracks.items():
        best_score = -1.0
        selected_entry = None
        for entry in entries:
            y1 = entry["bbox"][1]
            border_penalty = max(0.01, y1 / 80.0) if y1 < 80 else 1.0
            score = (0.50 * entry["conf"] + 0.30 * entry["centrality"] + 0.20 * entry["rel_height"]) * border_penalty
            if score > best_score:
                best_score = score
                selected_entry = entry
        if selected_entry:
            best_frame_indices[tid] = selected_entry["frame_idx"]
            best_metadata[tid] = selected_entry

    best_pole_results = {}
    if best_frame_indices:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        target_indices = set(best_frame_indices.values())
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_indices:
                for tid, idx in best_frame_indices.items():
                    if idx == frame_idx:
                        meta = best_metadata[tid]
                        best_pole_results[tid] = {
                            "frame": frame.copy(),
                            "timestamp_sec": meta["timestamp_sec"],
                            "bbox": meta["bbox"],
                            "conf": meta["conf"],
                            "frame_idx": idx,
                        }
            frame_idx += 1
        cap.release()

    return best_pole_results


class VideoWorker(threading.Thread):
    def __init__(self, name):
        super().__init__(name=name, daemon=True)
        self.pipeline_engine = None
        self.cable_model = None

    def run(self):
        print(f"[QueueWorker] Worker {self.name} started.")
        try:
            from pipeline import InfrastructurePipeline
            from ultralytics import YOLO

            self.pipeline_engine = InfrastructurePipeline(
                component_model_path="models/pole_best6.pt",
                insulator_model_path="models/insulator_new.pt",
                shed_model_path="models/shed_model.pt",
                crossarm_model_path="models/channel_12class_v2.pt"
            )
            self.cable_model = YOLO("models/cable_best.pt")
            print(f"[QueueWorker] Worker {self.name} successfully loaded isolated models.")
        except Exception as exc:
            print(f"[QueueWorker Error] Worker {self.name} failed loading models: {exc}")
            traceback.print_exc()
            return

        while True:
            try:
                task_id, video_id, target_fps = video_queue.get()
                print(f"[QueueWorker] Worker {self.name} picked up task {task_id}")
                temp_video_path = None

                with task_lock:
                    tasks[task_id] = {
                        "status": "processing",
                        "progress": 5,
                        "results": [],
                        "error": None,
                    }

                try:
                    video_row = get_uploaded_video(video_id)
                    if not video_row:
                        raise FileNotFoundError(f"Uploaded video {video_id} not found in database")

                    suffix = Path(video_row["original_filename"] or "video.mp4").suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(bytes(video_row["video_data"]))
                        temp_video_path = temp_file.name

                    results = self.process_video_task(task_id, temp_video_path, target_fps)

                    with task_lock:
                        tasks[task_id] = {
                            "status": "done",
                            "progress": 100,
                            "results": results,
                            "error": None,
                        }
                    print(f"[QueueWorker] Worker {self.name} completed task {task_id}")
                except Exception as exc:
                    print(f"[QueueWorker Error] Task {task_id} failed: {exc}")
                    traceback.print_exc()
                    with task_lock:
                        tasks[task_id] = {
                            "status": "failed",
                            "progress": 0,
                            "results": [],
                            "error": str(exc),
                        }
                finally:
                    if temp_video_path and os.path.exists(temp_video_path):
                        try:
                            os.remove(temp_video_path)
                            print(f"[QueueWorker] Cleaned up temp video {temp_video_path}")
                        except Exception as rm_err:
                            print(f"[QueueWorker Warning] Failed deleting {temp_video_path}: {rm_err}")

                    video_queue.task_done()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[QueueWorker Loop Error] {exc}")

    def process_video_task(self, task_id, video_path, target_fps):
        import cv2

        from image_processing import process_image_file

        self.update_progress(task_id, 10)

        def progress_callback(progress_value):
            self.update_progress(task_id, int(10 + (progress_value * 60)))

        best_poles = extract_best_pole_frames(
            video_path,
            self.pipeline_engine,
            target_fps=target_fps,
            progress_callback=progress_callback,
        )

        self.update_progress(task_id, 75)

        results = []
        total_poles = len(best_poles)

        for idx, (track_id, data) in enumerate(best_poles.items()):
            frame_img = data["frame"]
            _, buffer = cv2.imencode('.jpg', frame_img)
            file_stream = io.BytesIO(buffer.tobytes())

            pred_res = process_image_file(
                file_stream,
                pipeline_override=self.pipeline_engine,
                cable_model_override=self.cable_model,
            )
            if not any(is_pole_label(d.get('label')) and meets_pole_confidence(d.get('confidence')) for d in pred_res.get('detections', [])):
                self.update_progress(task_id, min(95, int(75 + ((idx + 1) / total_poles) * 20)))
                continue

            ins_count = sum(
                1 for det in pred_res.get("detections", [])
                if any(keyword in det["label"].lower() for keyword in ["insulator", "ins", "pin", "disc", "shackle"])
            )

            crossarm_shape = "none"
            for det in pred_res.get("detections", []):
                label = det["label"].lower()
                if any(keyword in label for keyword in ["v_cross_arm", "v_cross", "t_rising", "tapping_arm", "tapping_channel", "side_arm", "side_arm_channel", "top_cleat", "special_clamp", "stay_set", "ab_switch", "box_arm", "crossarm"]):
                    crossarm_shape = det["label"]
                    break

            results.append({
                "pole_id": track_id,
                "best_frame": f"best_pole_{track_id}_{uuid.uuid4().hex[:8]}.jpg",
                "timestamp_sec": round(data["timestamp_sec"], 2),
                "pole_type": pred_res.get("master", {}).get("pole_type", "unknown"),
                "channel_type": pred_res.get("master", {}).get("final_class", "none"),
                "crossarm_shape": crossarm_shape,
                "insulator_count": ins_count,
                "confidence": round(data["conf"], 2),
                "detections": pred_res.get("detections", []),
                "annotated_image": pred_res.get("annotated_image", ""),
                "master": pred_res.get("master", {}),
                "survey_questionnaire": pred_res.get("survey_questionnaire", {}),
                "width": pred_res.get("width", 0),
                "height": pred_res.get("height", 0),
            })

            step_progress = int(75 + ((idx + 1) / total_poles) * 20) if total_poles else 95
            self.update_progress(task_id, min(95, step_progress))

        self.update_progress(task_id, 98)
        return results

    def update_progress(self, task_id, progress_val):
        with task_lock:
            if task_id in tasks:
                tasks[task_id]["progress"] = progress_val


_workers_started = False
_worker_start_lock = threading.Lock()


def init_workers():
    global _workers_started
    with _worker_start_lock:
        if _workers_started:
            return
        # One worker avoids concurrently mutating shared YOLO predictors.
        worker = VideoWorker(name='VideoWorker')
        worker.start()
        _workers_started = True
        print('[QueueWorker] Background worker initialized successfully.')


def submit_task(video_id, target_fps=2):
    init_workers()
    task_id = str(uuid.uuid4())
    with task_lock:
        tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "results": [],
            "error": None,
        }
    video_queue.put((task_id, video_id, target_fps))
    return task_id


def get_task_status(task_id):
    with task_lock:
        return tasks.get(task_id)

