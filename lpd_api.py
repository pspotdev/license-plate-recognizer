import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
import onnxruntime as ort
from pyctcdecode import build_ctcdecoder
import json
import torch
from PIL import Image
import io
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import datetime
import re
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
import os
YOLO_PLATE_ONNX_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./best.onnx")
CRNN_ONNX_MODEL_PATH = os.getenv("CRNN_MODEL_PATH", "./resnet18_quantized.onnx")
PLATE_DICT_PATH = os.getenv("PLATE_DICT_PATH", "./plate_dict.json")
KENLM_MODEL_PATH = os.getenv("KENLM_MODEL_PATH", "./plate_lm13.arpa")
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
MAX_DET = 4
GAMMA = 1.0
IMG_WIDTH = 256
IMG_HEIGHT = 64
CONFIDENCE = 0.01
CHAR_LENGTH = 4
CHARSET = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"

# Load character set and decoder
try:
    with open(PLATE_DICT_PATH, "r", encoding="utf-8") as f:
        plate_dict = json.load(f)
except Exception as e:
    logger.error(f"Failed to load plate dictionary: {e}")
    raise Exception(f"Failed to load plate dictionary: {e}")

hotwords = []
for state, rto_list in plate_dict.items():
    hotwords.append(state)
    for rto in rto_list:
        hotwords.append(f"{state}{rto}")

labels = [""] + list(CHARSET)
decoder = build_ctcdecoder(
    labels,
    kenlm_model_path=KENLM_MODEL_PATH,
    alpha=0.6,
)

# Load YOLO OBB model
try:
    model = YOLO(YOLO_PLATE_ONNX_MODEL_PATH, task='obb')
    logger.info("YOLO OBB model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO OBB model: {e}")
    raise Exception(f"Failed to load YOLO OBB model: {e}")

# Load CRNN ONNX model
try:
    providers = ['CPUExecutionProvider']
    session_crnn = ort.InferenceSession(CRNN_ONNX_MODEL_PATH, providers=providers)
    crnn_input_name = session_crnn.get_inputs()[0].name
    logger.info(f"CRNN ONNX model loaded successfully with providers: {session_crnn.get_providers()}")
except Exception as e:
    logger.error(f"Failed to load CRNN ONNX model: {e}")
    raise Exception(f"Failed to load CRNN ONNX model: {e}")

# Helper Functions (unchanged from original code)
def four_point_transform(image, pts):
    rect = pts.astype("float32")
    s = rect.sum(axis=1)
    diff = np.diff(rect, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = rect[np.argmin(s)]
    ordered[2] = rect[np.argmax(s)]
    ordered[1] = rect[np.argmin(diff)]
    ordered[3] = rect[np.argmax(diff)]

    (tl, tr, br, bl) = ordered
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def non_max_suppression_obb(obbs, scores, iou_threshold):
    if len(obbs) == 0:
        return np.array([]), np.array([])
    
    polygons = [Polygon(obb.reshape(4, 2)) for obb in obbs]
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0 and len(keep) < MAX_DET:
        i = order[0]
        keep.append(i)
        ious = []
        for j in order[1:]:
            try:
                inter = polygons[i].intersection(polygons[j]).area
                union = polygons[i].area + polygons[j].area - inter
                iou = inter / union if union > 0 else 0.0
            except:
                iou = 0.0
            ious.append(iou)
        inds = np.where(np.array(ious) <= iou_threshold)[0]
        order = order[inds + 1]
    
    keep = np.array(keep)
    return keep, scores[keep]

def gamma_corr(img, gamma, c=1.0):
    img_norm = img / 255.0
    img_gamma = c * (img_norm ** gamma)
    img_gamma_scaled = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
    return img_gamma_scaled

def preprocess_crnn_image(img_array):
    img = Image.fromarray(img_array).convert('L')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img[np.newaxis, np.newaxis, :, :]
    return img

def decode_predictions_beam(logits):
    log_probs = torch.log_softmax(torch.from_numpy(logits), dim=2).numpy()
    return [
        decoder.decode(
            lp, beam_width=16, hotwords=hotwords, hotword_weight=5.0
        )
        for lp in log_probs
    ]

def correct_plate(predicted):
    plate = predicted.upper()
    corrected = list(plate)
    valid_states = set(plate_dict.keys())
    
    corrections = {
        '0': ['Q'], '1': ['L'], '2': ['Z'], '5': ['S'], '8': ['B'],
        'B': ['8'], 'G': ['6'], 'L': ['1'], 'S': ['5'], 'Z': ['2'],
        'D': ['0'], 'Q': ['0']
    }

    def suggest(char, expected):
        suggestions = corrections.get(char, [])
        if expected == 'alpha':
            return [c for c in suggestions if c.isalpha()]
        elif expected == 'digit':
            return [c for c in suggestions if c.isdigit()]
        return []

    if len(corrected) < 2 or not corrected[0].isalpha() or not corrected[1].isalpha():
        return plate

    state_code = corrected[0] + corrected[1]
    if state_code not in valid_states:
        for alt2 in suggest(corrected[1], 'alpha'):
            trial = corrected[0] + alt2
            if trial in valid_states:
                corrected[1] = alt2
                state_code = trial
                break

    rto_digits = ''
    if state_code == 'DL':
        if len(corrected) > 2 and corrected[2].isdigit():
            rto_digits = corrected[2]
            if len(corrected) > 3 and corrected[3].isdigit():
                rto_digits += corrected[3]
        elif len(corrected) > 2:
            for alt in suggest(corrected[2], 'digit'):
                corrected[2] = alt
                rto_digits = corrected[2]
                break
    else:
        for i in [2, 3]:
            if len(corrected) > i and not corrected[i].isdigit():
                for alt in suggest(corrected[i], 'digit'):
                    corrected[i] = alt
                    break
        rto_digits = ''.join(corrected[2:4])

    if state_code in plate_dict and rto_digits not in plate_dict[state_code]:
        pass

    if len(corrected) > 4 and not corrected[4].isalpha():
        for alt in suggest(corrected[4], 'alpha'):
            corrected[4] = alt
            break

    if len(corrected) > 5 and not corrected[5].isalpha():
        for alt in suggest(corrected[5], 'alpha'):
            corrected[5] = alt
            break

    if len(corrected) >= 5 and not corrected[-5].isalpha():
        for alt in suggest(corrected[-5], 'alpha'):
            corrected[-5] = alt
            break

    if len(corrected) >= 4:
        for i in range(-4, 0):
            if not corrected[i].isdigit():
                for alt in suggest(corrected[i], 'digit'):
                    corrected[i] = alt
                    break

    return ''.join(corrected)

def is_valid_plate_format(plate_text):
    if not plate_text or not isinstance(plate_text, str):
        return False
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{0,2}\d{1,4}$'
    return bool(re.match(pattern, plate_text.upper()))

def recognize_license_plate_crnn(plate_array, session_crnn, crnn_input_name):
    try:
        input_data = preprocess_crnn_image(plate_array)
        logits = session_crnn.run(None, {crnn_input_name: input_data})[0]
        log_probs = torch.log_softmax(torch.from_numpy(logits), dim=2)
        max_probs = torch.max(log_probs, dim=2)[0]
        valid_probs = max_probs[0][max_probs[0] != float('-inf')]
        confidence = torch.mean(valid_probs).item() if valid_probs.numel() > 0 else 0.0
        normalized_confidence = max(0.0, min((confidence + 5.0) * 20.0, 100.0))
        raw_prediction = decode_predictions_beam(logits)[0]
        plate_text = correct_plate(raw_prediction)
        if not is_valid_plate_format(plate_text):
            return "", 0.0
        return plate_text, normalized_confidence
    except Exception as e:
        logger.error(f"CRNN inference error: {e}")
        return "", 0.0

def extract_lpd(image_data: bytes, model, session_crnn, crnn_input_name) -> Dict:
    # Convert bytes to image
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            logger.error("Failed to decode image")
            return {"error": "Failed to decode image", "plates": []}
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return {"error": f"Image decoding error: {e}", "plates": []}

    results = model.predict(source=frame, conf=CONF_THRESHOLD, stream=False, verbose=False)
    annotated_frame = frame.copy()
    cropped_plates = []
    plate_texts = []

    if hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb.xyxyxyxy) > 0:
        obbs = results[0].obb.xyxyxyxy
        confs = results[0].obb.conf if hasattr(results[0].obb, "conf") else None

        obbs_np = obbs.cpu().numpy() if hasattr(obbs, 'cpu') else obbs
        confs_np = confs.cpu().numpy() if confs is not None else np.zeros(len(obbs_np))

        # Apply NMS
        keep_indices, keep_scores = non_max_suppression_obb(obbs_np, confs_np, IOU_THRESHOLD)
        obbs_np = obbs_np[keep_indices]
        confs_np = keep_scores

        for i, obb in enumerate(obbs_np):
            pts = obb.reshape((4, 2))
            if np.any(np.isnan(pts)) or np.any(pts == None):
                logger.warning(f"Invalid points in OBB {i}")
                continue

            # Draw OBB on annotated frame
            cv2.polylines(annotated_frame, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

            # Crop and warp the plate
            license_plate_bgr = four_point_transform(frame, pts)
            license_plate_bgr = gamma_corr(license_plate_bgr, gamma=GAMMA)
            if license_plate_bgr.size == 0:
                continue

            # Recognize license plate text
            plate_text, confidence = recognize_license_plate_crnn(cv2.cvtColor(license_plate_bgr, cv2.COLOR_BGR2GRAY), session_crnn, crnn_input_name)
            if plate_text and confidence >= CONFIDENCE and len(plate_text) >= CHAR_LENGTH:
                # Convert cropped plate to base64
                _, buffer = cv2.imencode('.jpg', license_plate_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                cropped_plates.append(cropped_base64)
                plate_texts.append({
                    'text': plate_text,
                    'confidence': confidence,
                    'cropped_image': cropped_base64
                })
                cv2.putText(annotated_frame, f"Plate: {plate_text}", 
                            (int(pts[0, 0]), int(pts[0, 1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                logger.info(f"Plate recognition skipped: {plate_text}, confidence: {confidence:.2f}")

    # Convert annotated frame to base64
    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "plates": plate_texts,
        "annotated_image": annotated_base64,
        "timestamp": datetime.utcnow().isoformat()
    }

# FastAPI app
app = FastAPI(title="License Plate Detection and Recognition API")

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    """
    Process an uploaded image for license plate detection and recognition.
    Returns detected plates with text, confidence, cropped images, and annotated image.
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        # Read image bytes
        image_data = await file.read()
        # Process image
        result = extract_lpd(image_data, model, session_crnn, crnn_input_name)
        return result
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)