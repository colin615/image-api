# main.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os, shutil, uuid, io
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2

# optional rembg (only import if available)
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

app = FastAPI()

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

def read_image_bytes(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def save_image(img_bgr, path):
    cv2.imwrite(path, img_bgr)

def order_points_clockwise(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points_clockwise(pts)
    (tl, tr, br, bl) = rect
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
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_document_contour(img_bgr):
    img = img_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2)
            return pts
    return None

def deep_dewarp_fallback(img_bgr):
    """
    Placeholder: if you add SuperPoint+SuperGlue or a dewarping network,
    implement inference here and return a flattened BGR image.
    For CPU-first setup, we keep this a no-op (return original).
    """
    return img_bgr

def remove_background_if_needed(img_bgr, use_rembg=False):
    if use_rembg and REMBG_AVAILABLE:
        # rembg expects bytes or PIL/numpy; we'll pass numpy
        out = rembg_remove(img_bgr)
        # rembg returns RGBA or numpy with alpha; ensure BGR output with white bg
        if isinstance(out, bytes):
            arr = np.frombuffer(out, np.uint8)
            pil = Image.open(io.BytesIO(arr)).convert("RGBA")
        else:
            pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGBA))
        bg = Image.new("RGBA", pil.size, (255,255,255,255))
        bg.paste(pil, (0,0), pil)
        pil_rgb = bg.convert("RGB")
        return cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    return img_bgr

def enhance_image(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil = ImageOps.autocontrast(pil, cutoff=0)
    pil = pil.filter(ImageFilter.SHARPEN)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...),
                        high_quality: bool = Query(False),
                        use_rembg: bool = Query(False)):
    raw = await file.read()
    img = read_image_bytes(raw)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Couldn't decode image."})

    method = "opencv_quad"
    quad = find_document_contour(img)
    if quad is not None:
        warped = four_point_transform(img, quad)
    else:
        warped = deep_dewarp_fallback(img)
        method = "deep_fallback"

    # Optional background removal (use_rembg query param) - fairly slow on CPU for big images
    warped = remove_background_if_needed(warped, use_rembg=use_rembg)

    # Enhancement if requested
    if high_quality:
        warped = enhance_image(warped)
        method += "+enhance"

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    new_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, new_filename)
    save_image(warped, file_path)

    file_url = f"/uploads/{new_filename}"
    return JSONResponse(content={"file_url": file_url, "method": method})
