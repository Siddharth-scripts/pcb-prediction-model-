"""
PCB Component Detection Backend API
- Loads YOLO model (best.pt), runs inference on images
- Returns total component count (bounding boxes) and PCB classification
- PCB = True if at least one component is detected/classified
"""

import os
import io
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Model next to app.py (works when run from D:\backend or any USB path)
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "best.pt"
STATIC_DIR = BACKEND_DIR / "static"

app = FastAPI(
    title="PCB Component Detection API",
    description="Detect PCB components with YOLO and count bounding boxes. Image is classified as PCB if at least one component is detected.",
    version="1.0.0",
)

# Load model once at startup
model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH}, classes: {model.names}")


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    total_components: int
    is_pcb: bool
    component_counts: dict
    detections: List[BoundingBox]
    class_names: dict
    annotated_image_b64: Optional[str] = None


@app.get("/")
def index():
    """Serve the PCB detection upload page (shows annotated image with bounding boxes)."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Static page not found")
    return FileResponse(index_path, media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "path": str(MODEL_PATH),
        "names": model.names,
        "num_classes": len(model.names),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.25,
    imgsz: int = 640,
    return_annotated_image: bool = False,
):
    """
    Run detection on an uploaded image.
    - total_components: number of bounding boxes (detected components)
    - is_pcb: True if at least one component is detected (image classified as PCB)
    - component_counts: count per class (e.g. Resistor, Capacitor, IC)
    - detections: list of boxes with class and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Allow common image types
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type and file.content_type.lower() not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    import numpy as np
    import cv2

    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Inference
    results = model.predict(
        source=img,
        conf=conf,
        imgsz=imgsz,
        device="cpu",
        verbose=False,
    )
    r = results[0]

    # Build detections and counts
    detections = []
    component_counts = {name: 0 for name in model.names.values()}

    if r.boxes is not None and r.boxes.data is not None:
        for box in r.boxes:
            xy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
            cls_id = int(box.cls[0].cpu())
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0].cpu())

            detections.append(
                BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=confidence,
                )
            )
            component_counts[cls_name] = component_counts.get(cls_name, 0) + 1

    total_components = len(detections)
    # PCB: at least one component classified => image is a PCB
    is_pcb = total_components >= 1

    out = {
        "total_components": total_components,
        "is_pcb": is_pcb,
        "component_counts": component_counts,
        "detections": detections,
        "class_names": model.names,
        "annotated_image_b64": None,
    }

    # Always return annotated image when requested (boxes drawn if any detections)
    if return_annotated_image:
        annot = img.copy()
        for d in detections:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annot, f"{d.class_name} {d.confidence:.2f}",
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        _, buf = cv2.imencode(".png", annot)
        out["annotated_image_b64"] = base64.b64encode(buf.tobytes()).decode("utf-8")

    return out


@app.post("/predict-path")
async def predict_from_path(
    image_path: str,
    conf: float = 0.25,
    imgsz: int = 640,
):
    """Run detection on an image path (e.g. for local testing)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import cv2
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / image_path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=400, detail="Could not load image")

    results = model.predict(source=img, conf=conf, imgsz=imgsz, device="cpu", verbose=False)
    r = results[0]

    detections = []
    component_counts = {name: 0 for name in model.names.values()}

    if r.boxes is not None and r.boxes.data is not None:
        for box in r.boxes:
            xy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu())
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0].cpu())
            detections.append({
                "x1": float(xy[0]), "y1": float(xy[1]), "x2": float(xy[2]), "y2": float(xy[3]),
                "class_id": cls_id, "class_name": cls_name, "confidence": confidence,
            })
            component_counts[cls_name] = component_counts.get(cls_name, 0) + 1

    total_components = len(detections)
    is_pcb = total_components >= 1

    return {
        "total_components": total_components,
        "is_pcb": is_pcb,
        "component_counts": component_counts,
        "detections": detections,
        "class_names": model.names,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
