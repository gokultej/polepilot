import base64
import os
import tempfile
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from config import filter_pole_detections, THRESHOLD_POLE

from detection_processing import (
    build_master_data,
    build_survey_questionnaire,
    collect_model_detections,
    extract_conductor_detections,
)
from pipeline import InfrastructurePipeline

pipeline_engine = InfrastructurePipeline(
    component_model_path="models/pole_best6.pt",
    insulator_model_path="models/insulator_new.pt",
    shed_model_path="models/shed_model.pt",
    crossarm_model_path="models/channel_12class_v2.pt",
)

cable_model = YOLO("models/cable_best.pt")


def _run_raw_pipeline(temp_filename, log_mem, active_pipeline_engine=None):
    import gc

    log_mem("Before Pipeline")
    engine = active_pipeline_engine or pipeline_engine
    pipe_res = engine.predict(temp_filename)
    log_mem("After Pipeline")
    gc.collect()
    return pipe_res


def _build_cable_mask(temp_filename, height, width, log_mem, active_cable_model=None):
    mask_resized = np.zeros((height, width), dtype=np.uint8)
    try:
        cable_start = time.perf_counter()
        model = active_cable_model or cable_model
        cable_results = model(temp_filename, imgsz=1024, conf=0.10, verbose=False)
        if cable_results and cable_results[0].masks is not None and cable_results[0].masks.xy is not None:
            for poly_pts in cable_results[0].masks.xy:
                if len(poly_pts) > 0:
                    cv2.fillPoly(mask_resized, [poly_pts.astype(np.int32)], 255)
        log_mem("After Cable YOLO")
        print(f"[Stage Timing] cable_yolo={time.perf_counter() - cable_start:.3f}s")
    except Exception as exc:
        print(f"[Cable Inference Error] {exc}")
    return mask_resized


def _should_run_conductor_stage(final_detections, main_pole_boxes):
    if not main_pole_boxes:
        print("[Conductor Skip] No main pole box available after pipeline.")
        return False
    structural_labels = {"crossarm", "v_cross_arm", "v_cross", "t_rising", "t_raising", "tapping_arm", "tapping_channel", "side_arm", "side_arm_channel", "top_cleat", "special_clamp", "stay_set", "ab_switch", "box_arm", "insulator", "ht_pin", "ht_disc", "lt_pin", "shackle_insulator", "street_light", "lamp_head", "dtr", "dtr_tank"}
    labels = {str(det.get("label", "")).lower() for det in final_detections}
    if not labels.intersection(structural_labels):
        print("[Conductor Skip] No structural hardware found near the accepted pole.")
        return False
    return True


def _prepare_conductor_masks(mask_resized):
    from skimage.morphology import skeletonize

    prep_start = time.perf_counter()
    source_mask = mask_resized
    scale = 1.0
    max_dim = max(mask_resized.shape[:2])
    if max_dim > 1280:
        scale = 1280.0 / max_dim
        scaled_size = (max(1, int(mask_resized.shape[1] * scale)), max(1, int(mask_resized.shape[0] * scale)))
        source_mask = cv2.resize(mask_resized, scaled_size, interpolation=cv2.INTER_AREA)

    dist_small = cv2.distanceTransform(source_mask, cv2.DIST_L2, 5)
    skel_small = (skeletonize(source_mask / 255.0) > 0).astype(np.uint8)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(source_mask, dilate_kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed_small = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, close_kernel)

    if scale != 1.0:
        target_size = (mask_resized.shape[1], mask_resized.shape[0])
        dist = cv2.resize(dist_small, target_size, interpolation=cv2.INTER_LINEAR) / scale
        skel = cv2.resize(skel_small, target_size, interpolation=cv2.INTER_NEAREST)
        mask_closed = cv2.resize(mask_closed_small, target_size, interpolation=cv2.INTER_NEAREST)
    else:
        dist = dist_small
        skel = skel_small
        mask_closed = mask_closed_small

    print(f"[Stage Timing] conductor_mask_prep={time.perf_counter() - prep_start:.3f}s (scale={scale:.3f})")
    return dist, skel, mask_closed


def process_image_file(file_stream, pipeline_override=None, cable_model_override=None):
    import gc
    import psutil

    def log_mem(step):
        memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"[Memory] {step}: {memory_mb:.1f} MB")

    log_mem("Start Inference")
    temp_filename = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file_stream.read())
            temp_filename = temp_file.name

        pipe_res = _run_raw_pipeline(temp_filename, log_mem, active_pipeline_engine=pipeline_override)
        img = cv2.imread(temp_filename)
        height, width = img.shape[:2]

        final_detections, hardware_mask, main_pole_boxes = collect_model_detections(pipe_res, width, height)
        merged_conductors = []

        if not _should_run_conductor_stage(final_detections, main_pole_boxes):
            log_mem("Skipped conductor extraction")
        else:
            mask_resized = _build_cable_mask(temp_filename, height, width, log_mem, active_cable_model=cable_model_override)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            conductor_extract_start = time.perf_counter()
            dist, skel, mask_closed = _prepare_conductor_masks(mask_resized)
            merged_conductors = extract_conductor_detections(mask_closed, hardware_mask, main_pole_boxes, dist, skel, width, height)
            print(f"[Stage Timing] conductor_extract={time.perf_counter() - conductor_extract_start:.3f}s, count={len(merged_conductors)}")
            final_detections.extend(merged_conductors)

        _, buffer = cv2.imencode('.jpg', img)
        return {
            "detections": filter_pole_detections(final_detections),
            "pole_confidence_threshold": THRESHOLD_POLE,
            "master": build_master_data(pipe_res),
            "survey_questionnaire": build_survey_questionnaire(pipe_res, merged_conductors),
            "annotated_image": base64.b64encode(buffer).decode('utf-8'),
            "width": width,
            "height": height,
            "inference_mode": "accurate",
        }
    finally:
        if 'img' in locals():
            del img
        if 'pipe_res' in locals():
            del pipe_res
        if 'mask_resized' in locals():
            del mask_resized
        if 'hardware_mask' in locals():
            del hardware_mask
        if 'mask_closed' in locals():
            del mask_closed
        if 'dist' in locals():
            del dist
        if 'skel' in locals():
            del skel
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        log_mem("Request End")
