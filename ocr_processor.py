import os
import io
import json
import base64
import re
import urllib.request
import urllib.error
from html import escape
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen3-vl-30b" 

IMAGE_FOLDER = Path("./images")
TEXT_FOLDER = Path("./texts")
KEYWORDS_FOLDER = Path("./keywords")
ORIENTATION_FOLDER = Path("./orientations")
MASTER_KEYWORDS_FILE = Path("./master_keywords.json")
HTML_OUTPUT = Path("./index.html")
ALIGNED_IMAGE_FOLDER = Path("./docs/.aligned_images")

IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
TEXT_FOLDER.mkdir(parents=True, exist_ok=True)
KEYWORDS_FOLDER.mkdir(parents=True, exist_ok=True)
ORIENTATION_FOLDER.mkdir(parents=True, exist_ok=True)
ALIGNED_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXIF_ORIENTATION_TAG = 274
ORIENTATION_MODEL_VERSION = 2


def iter_image_files():
    return sorted(
        [image_path for image_path in IMAGE_FOLDER.rglob("*") if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda image_path: image_path.relative_to(IMAGE_FOLDER).as_posix().lower(),
    )


def relative_image_info(image_path):
    relative_image_path = image_path.relative_to(IMAGE_FOLDER)
    group_path = relative_image_path.parent
    group_key = "" if group_path == Path(".") else group_path.as_posix()
    group_label = "none" if not group_key else group_key
    group_id = "group-none" if not group_key else "group-" + group_key.replace("/", "-")
    return relative_image_path, group_key, group_label, group_id


def mirrored_text_path(relative_image_path):
    return TEXT_FOLDER / relative_image_path.with_suffix(".txt")


def mirrored_keyword_path(relative_image_path):
    return KEYWORDS_FOLDER / relative_image_path.with_suffix(".json")


def mirrored_orientation_path(relative_image_path):
    return ORIENTATION_FOLDER / relative_image_path.with_suffix(".json")


def discover_image_files():
    return sorted(
        (
            image_path
            for image_path in IMAGE_FOLDER.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(IMAGE_FOLDER).as_posix().lower(),
    )


def slugify(value):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "none"


def relative_image_path_for(image_path):
    return image_path.relative_to(IMAGE_FOLDER)


def group_label_for(relative_image_path):
    relative_group = relative_image_path.parent.as_posix()
    return "None" if relative_group in ("", ".") else relative_group


def output_path_for(root_folder, relative_image_path, new_suffix):
    return root_folder / relative_image_path.with_suffix(new_suffix)

def fix_image_rotation(image_path):
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            orientation = exif.get(EXIF_ORIENTATION_TAG, 1) if exif else 1
            if orientation != 1:
                corrected_img = ImageOps.exif_transpose(img)
                corrected_exif = corrected_img.getexif()
                corrected_exif[EXIF_ORIENTATION_TAG] = 1
                corrected_img.save(image_path, format=img.format, exif=corrected_exif.tobytes())
                print(f"Fixed alignment for: {image_path.name}")
    except Exception as e:
        print(f"Could not auto-rotate {image_path.name}: {e}")


def parse_rotation_response(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "rotation" in parsed:
            value = int(parsed["rotation"])
            if value in (0, 90, 180, 270):
                return value
        if isinstance(parsed, int) and parsed in (0, 90, 180, 270):
            return parsed
    except Exception:
        pass
    match = re.search(r"\b(0|90|180|270)\b", cleaned)
    if match:
        return int(match.group(1))
    return 0


def detect_display_rotation(image_path):
    try:
        with Image.open(image_path) as img:
            normalized = ImageOps.exif_transpose(img)
            variants = {}
            for rotation in (0, 90, 180, 270):
                candidate = normalized if rotation == 0 else normalized.rotate(-rotation, expand=True)
                variants[rotation] = encode_pil_image_to_base64(candidate, image_format="JPEG")
    except Exception as e:
        print(f"Warning: Could not prepare rotation variants for {image_path.name}: {e}")
        return 0

    prompt = (
        "You are comparing four rotated versions of the same handwritten image. "
        "Choose the clockwise rotation where text is upright and easiest to read left-to-right with horizontal lines. "
        "If there are two pages in one image, both pages should appear upright as a natural landscape spread. "
        "Allowed values: 0, 90, 180, 270. "
        "Output only compact JSON in this format: {\"rotation\": 90}."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Variant A: 0 degrees clockwise"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{variants[0]}"}},
                {"type": "text", "text": "Variant B: 90 degrees clockwise"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{variants[90]}"}},
                {"type": "text", "text": "Variant C: 180 degrees clockwise"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{variants[180]}"}},
                {"type": "text", "text": "Variant D: 270 degrees clockwise"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{variants[270]}"}},
            ],
        }
    ]
    try:
        response_text = send_local_llm_request(messages, temperature=0.0)
        return parse_rotation_response(response_text)
    except Exception as e:
        print(f"Warning: Could not determine rotation for {image_path.name}: {e}")
        return 0


def create_aligned_display_image(image_path, relative_image_path, clockwise_rotation):
    aligned_output = ALIGNED_IMAGE_FOLDER / relative_image_path
    aligned_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(image_path) as img:
            aligned = ImageOps.exif_transpose(img)
            if clockwise_rotation in (90, 180, 270):
                aligned = aligned.rotate(-clockwise_rotation, expand=True)
            save_format = img.format or "JPEG"
            aligned.save(aligned_output, format=save_format)
    except Exception as e:
        print(f"Warning: Failed to create aligned display image for {image_path.name}: {e}")
        return image_path
    return aligned_output

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(image_obj, image_format="JPEG"):
    buffer = io.BytesIO()
    image_obj.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def send_local_llm_request(messages, temperature=0.1):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LM_STUDIO_URL, 
        data=data, 
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            return res_body["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        raise ConnectionError(f"Could not connect to LM Studio at {LM_STUDIO_URL}: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise ValueError(f"Unexpected response structure: {e}")

def perform_ocr(image_path):
    base64_image = encode_image_to_base64(image_path)
    prompt = (
        "You are an expert handwriting transcription AI. "
        "Transcribe all text from this image perfectly. "
        "The text is written on lined paper. Ignore the background lines entirely. "
        "Maintain the paragraph structure, line breaks, and formatting of the original handwriting. "
        "Do not add any commentary, explanations, or markdown boxes. Output ONLY the transcribed text."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    try:
        return send_local_llm_request(messages, temperature=0.1)
    except Exception as e:
        return f"ERROR processing image: {str(e)}"

def extract_keywords(transcribed_text):
    prompt = (
        "Analyze the following transcribed text and extract the most critical unique keywords, "
        "topics, names, or themes. Output the result strictly as a clean JSON array of strings. "
        "Example output formatting: [\"keyword1\", \"keyword2\", \"keyword3\"] "
        "Do not include any introductory remarks, explanations, or markdown code block markers. "
        f"Text to analyze:\n\n{transcribed_text}"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        raw_output = send_local_llm_request(messages, temperature=0.3).strip()
        if "```" in raw_output:
            parts = raw_output.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") and part.endswith("]"):
                    raw_output = part
                    break
        return json.loads(raw_output)
    except Exception as e:
        print(f"Warning: Failed to extract keywords cleanly: {e}")
        return ["Error extracting keywords"]

def build_html_page(processed_data):
    page_data = json.dumps(processed_data, ensure_ascii=False)
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Andrews Meanderings</title>
    <script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js"></script>
    <style>
        :root {
            --bg: #f3f6f2;
            --text: #1d2a1f;
            --panel: #ffffff;
            --accent: #2f6f4f;
            --accent-2: #6f9e7f;
            --line: #dbe6dd;
        }
        * { box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        body { font-family: "Trebuchet MS", "Segoe UI", sans-serif; margin: 24px; background: radial-gradient(circle at top, #edf7ef, var(--bg)); color: var(--text); }
        .page-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
        .page-title { display: flex; align-items: center; gap: 10px; }
        .header-thumb { width: 44px; height: 44px; border-radius: 8px; border: 1px solid #cfe0d3; object-fit: cover; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12); }
        h1 { margin: 0; color: #122017; letter-spacing: 0.4px; font-size: 2rem; }
        .header-tools { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .group-nav, .action-btn { border: 0; border-radius: 8px; background: linear-gradient(120deg, var(--accent), var(--accent-2)); color: #fff; padding: 10px 14px; font-size: 14px; font-weight: 700; }
        .group-nav { max-width: 240px; }
        .group-nav option { background: #ffffff; color: #1d2a1f; }
        .action-btn { cursor: pointer; }
        .action-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .complete-toggle { min-width: 150px; }
        .report { display: flex; flex-direction: column; gap: 18px; padding-bottom: 40px; }
        .group-section { background: rgba(255, 255, 255, 0.72); border: 1px solid rgba(219, 230, 221, 0.85); border-radius: 14px; box-shadow: 0 12px 30px rgba(14, 31, 18, 0.08); overflow: hidden; }
        .group-header { padding: 14px 18px; background: linear-gradient(120deg, rgba(47, 111, 79, 0.12), rgba(111, 158, 127, 0.18)); border-bottom: 1px solid rgba(219, 230, 221, 0.85); font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: #21382a; }
        .group-items { display: flex; flex-direction: column; gap: 18px; padding: 18px; }
        .image-entry { display: grid; grid-template-columns: minmax(280px, 44%) 1fr; gap: 18px; background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: 0 6px 18px rgba(9, 22, 13, 0.05); position: relative; }
        .image-entry.complete { outline: 2px solid rgba(47, 111, 79, 0.18); }
        .hide-complete .image-entry.complete { display: none; }
        .image-panel { position: relative; }
        .file-meta-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 10px; }
        .complete-checkbox { display: inline-flex; align-items: center; gap: 6px; padding: 6px 8px; border-radius: 999px; background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(219, 230, 221, 0.95); font-size: 12px; font-weight: 700; color: #284235; }
        .complete-checkbox input { margin: 0; }
        .image-preview-wrap { display: flex; width: 100%; justify-content: center; align-items: center; border-radius: 10px; background: #f3f7f4; min-height: 260px; padding: 10px; overflow: visible; }
        .text-editor { width: 100%; min-height: 180px; border: 1px solid var(--line); border-radius: 8px; padding: 14px; font-size: 15px; line-height: 1.6; font-family: inherit; white-space: pre-wrap; resize: none; box-sizing: border-box; }
        .image-preview { width: auto; max-width: 100%; height: auto; max-height: 70vh; display: block; object-fit: contain; image-orientation: from-image; border: 1px solid #ddd; border-radius: 10px; transition: transform 0.2s ease; transform-origin: center center; }
        .filename-header { font-weight: bold; font-size: 14px; color: #333; }
        .keyword-badge { display: inline-block; background: #e0f2fe; color: #0369a1; font-size: 12px; font-weight: 600; padding: 4px 8px; margin: 2px; border-radius: 12px; }
        .keyword-container { background: #f8fafc; padding: 10px 14px; border: 1px solid #dfe8e1; border-radius: 10px; margin: 0 0 14px; }
        .record-btn, .play-btn, .rotate-select, .zoom-btn { border: 0; border-radius: 6px; padding: 8px 12px; color: #fff; font-weight: 700; cursor: pointer; }
        .record-btn { background: #c1352a; }
        .record-btn.recording { background: #8f241f; }
        .play-btn { background: #2f6f4f; }
        .play-btn[hidden] { display: none; }
        .rotate-select { background: #274b78; }
        .zoom-btn { background: #51626f; min-width: 40px; padding: 8px 10px; }
        .recording-status { font-size: 13px; color: #425447; }
        .scroll-top-btn { position: fixed; right: 16px; bottom: 16px; z-index: 10000; border: 0; border-radius: 999px; padding: 10px 14px; background: #1f6b44; color: #fff; font-weight: 700; cursor: pointer; box-shadow: 0 10px 24px rgba(0, 0, 0, 0.2); display: none; }
        .scroll-top-btn.visible { display: inline-block; }
        .empty-state { padding: 18px; color: #4b6150; font-style: italic; }
        @media (max-width: 900px) { body { margin: 16px; } .page-header { align-items: flex-start; } .image-entry { grid-template-columns: 1fr; } .text-editor { min-height: 220px; } }
    </style>
</head>
<body>
    <div class="page-header">
        <div class="page-title">
            <img id="header-thumb" class="header-thumb" alt="Header image" src="andrew.png">
            <h1>Andrews Meanderings</h1>
        </div>
        <div class="header-tools">
            <select id="group-nav" class="group-nav" aria-label="Jump to group"></select>
            <button id="toggle-complete" class="action-btn complete-toggle" type="button">Hide Complete</button>
            <button id="download-all-recordings" class="action-btn" type="button">Download Texts + Recordings (.zip)</button>
        </div>
    </div>
    <div id="report" class="report"></div>
    <button id="scroll-top-btn" class="scroll-top-btn" type="button" aria-label="Scroll to top">Scroll to Top</button>

    <script>
        (function () {
            const TEXT_KEY_PREFIX = "ocr_text_";
            const AUDIO_KEY_PREFIX = "ocr_audio_";
            const COMPLETE_KEY_PREFIX = "ocr_complete_";
            const DISPLAY_ROTATION_KEY_PREFIX = "ocr_display_rotation_";
            const SHOW_COMPLETE_KEY = "ocr_show_complete";
            const RECORDING_TYPE = "audio/webm";
            const pageData = __PAGE_DATA__;
            const report = document.getElementById("report");
            const groupNav = document.getElementById("group-nav");
            const headerThumb = document.getElementById("header-thumb");
            const downloadButton = document.getElementById("download-all-recordings");
            const toggleCompleteButton = document.getElementById("toggle-complete");
            const scrollTopButton = document.getElementById("scroll-top-btn");
            const groups = new Map();
            const groupOrder = [];

            function storageKey(prefix, fileKey) {
                return prefix + fileKey;
            }

            function resizeEditor(editor) {
                editor.style.height = "auto";
                editor.style.height = `${editor.scrollHeight}px`;
            }

            function isShowingComplete() {
                return localStorage.getItem(SHOW_COMPLETE_KEY) !== "false";
            }

            function applyCompleteVisibility() {
                document.body.classList.toggle("hide-complete", !isShowingComplete());
                toggleCompleteButton.textContent = isShowingComplete() ? "Hide Complete" : "Show Complete";
            }

            function updateCompleteState(entry, suppressHideUpdate = false) {
                const isComplete = entry.completeCheckbox.checked;
                entry.element.classList.toggle("complete", isComplete);
                localStorage.setItem(storageKey(COMPLETE_KEY_PREFIX, entry.fileKey), isComplete ? "true" : "false");
                if (!suppressHideUpdate) {
                    entry.element.classList.toggle("hidden-by-toggle", !isShowingComplete() && isComplete);
                }
            }

            function updatePlayButtonLabel(audio, button) {
                button.textContent = audio.paused ? "Play" : "Stop";
            }

            function applyImageTransform(image, degrees, zoom) {
                image.style.transform = `rotate(${degrees}deg) scale(${zoom})`;
            }

            function dataUrlToBlob(dataUrl) {
                const parts = dataUrl.split(",");
                const mime = parts[0].match(/:(.*?);/)[1];
                const bytes = atob(parts[1]);
                const arr = new Uint8Array(bytes.length);
                for (let i = 0; i < bytes.length; i += 1) {
                    arr[i] = bytes.charCodeAt(i);
                }
                return new Blob([arr], { type: mime });
            }

            function toZipPath(folderName, relativeImagePath, extension) {
                return `${folderName}/${relativeImagePath.replace(/\\.[^.]+$/, extension)}`;
            }

            function buildEntry(item) {
                const entry = document.createElement("article");
                entry.className = "image-entry";
                entry.dataset.groupId = item.groupId;
                entry.dataset.fileKey = item.relativeImagePath;

                const imagePanel = document.createElement("div");
                imagePanel.className = "image-panel";

                const completeLabel = document.createElement("label");
                completeLabel.className = "complete-checkbox";

                const completeCheckbox = document.createElement("input");
                completeCheckbox.type = "checkbox";
                completeCheckbox.dataset.completeCheckbox = "true";
                completeCheckbox.checked = localStorage.getItem(storageKey(COMPLETE_KEY_PREFIX, item.relativeImagePath)) === "true";

                const completeText = document.createElement("span");
                completeText.textContent = "Complete";
                completeLabel.appendChild(completeCheckbox);
                completeLabel.appendChild(completeText);

                const imageWrap = document.createElement("div");
                imageWrap.className = "image-preview-wrap";

                const image = document.createElement("img");
                image.className = "image-preview";
                image.src = `${item.displayImagePath}?v=${item.modifiedTime}`;
                image.alt = item.filename;
                image.loading = "lazy";
                imageWrap.appendChild(image);

                imagePanel.appendChild(imageWrap);

                const textPane = document.createElement("div");
                textPane.className = "text-pane";

                const recordButton = document.createElement("button");
                recordButton.className = "record-btn";
                recordButton.type = "button";
                recordButton.textContent = "Record";

                const playButton = document.createElement("button");
                playButton.className = "play-btn";
                playButton.type = "button";
                playButton.textContent = "Play";

                const rotationSelect = document.createElement("select");
                rotationSelect.className = "rotate-select";
                rotationSelect.setAttribute("aria-label", "Rotate image");
                [0, 90, 180, 270].forEach((rotationOption) => {
                    rotationSelect.appendChild(new Option(`Rotate ${rotationOption}\u00b0`, String(rotationOption)));
                });

                const zoomOutButton = document.createElement("button");
                zoomOutButton.className = "zoom-btn";
                zoomOutButton.type = "button";
                zoomOutButton.textContent = "-";
                zoomOutButton.setAttribute("aria-label", "Zoom out image");

                const zoomInButton = document.createElement("button");
                zoomInButton.className = "zoom-btn";
                zoomInButton.type = "button";
                zoomInButton.textContent = "+";
                zoomInButton.setAttribute("aria-label", "Zoom in image");

                const status = document.createElement("span");
                status.className = "recording-status";
                status.textContent = "No recording";

                const audio = document.createElement("audio");
                audio.preload = "none";

                const filenameHeader = document.createElement("div");
                filenameHeader.className = "filename-header";
                filenameHeader.textContent = `File: ${item.relativeImagePath}`;

                const fileMetaRow = document.createElement("div");
                fileMetaRow.className = "file-meta-row";
                fileMetaRow.appendChild(filenameHeader);
                fileMetaRow.appendChild(completeLabel);
                fileMetaRow.appendChild(recordButton);
                fileMetaRow.appendChild(playButton);
                fileMetaRow.appendChild(rotationSelect);
                fileMetaRow.appendChild(zoomOutButton);
                fileMetaRow.appendChild(zoomInButton);
                fileMetaRow.appendChild(status);

                const storedRotationValue = localStorage.getItem(storageKey(DISPLAY_ROTATION_KEY_PREFIX, item.relativeImagePath));
                const savedRotation = storedRotationValue === null ? NaN : parseInt(storedRotationValue, 10);
                const defaultRotation = Number(item.rotation || 0);
                const initialRotation = [0, 90, 180, 270].includes(savedRotation)
                    ? savedRotation
                    : ([0, 90, 180, 270].includes(defaultRotation) ? defaultRotation : 0);
                const zoomKey = storageKey("ocr_zoom_", item.relativeImagePath);
                const storedZoom = parseFloat(localStorage.getItem(zoomKey) || "1");
                const initialZoom = Number.isFinite(storedZoom) ? Math.min(2.5, Math.max(0.5, storedZoom)) : 1;
                rotationSelect.value = String(initialRotation);
                applyImageTransform(image, initialRotation, initialZoom);

                const keywordContainer = document.createElement("div");
                keywordContainer.className = "keyword-container";
                const keywordLabel = document.createElement("strong");
                keywordLabel.textContent = "Keywords:";
                keywordContainer.appendChild(keywordLabel);
                if (item.keywords.length) {
                    item.keywords.forEach((keyword) => {
                        const badge = document.createElement("span");
                        badge.className = "keyword-badge";
                        badge.textContent = keyword;
                        keywordContainer.appendChild(badge);
                    });
                } else {
                    keywordContainer.appendChild(document.createTextNode(" None"));
                }

                const editor = document.createElement("textarea");
                editor.className = "text-editor";
                editor.dataset.textEditor = "true";
                editor.dataset.fileKey = item.relativeImagePath;
                editor.value = localStorage.getItem(storageKey(TEXT_KEY_PREFIX, item.relativeImagePath)) ?? item.text;

                textPane.appendChild(fileMetaRow);
                textPane.appendChild(keywordContainer);
                textPane.appendChild(editor);

                const existingRecording = localStorage.getItem(storageKey(AUDIO_KEY_PREFIX, item.relativeImagePath));
                if (existingRecording) {
                    audio.src = existingRecording;
                    playButton.hidden = false;
                    status.textContent = "Recording saved";
                } else {
                    playButton.hidden = true;
                }
                textPane.appendChild(audio);

                entry.appendChild(imagePanel);
                entry.appendChild(textPane);

                const entryRef = {
                    element: entry,
                    fileKey: item.relativeImagePath,
                    completeCheckbox,
                    recordButton,
                    playButton,
                    audio,
                    status,
                    editor,
                    image,
                    rotationSelect,
                    zoomLevel: initialZoom,
                };

                setTimeout(() => resizeEditor(editor), 0);
                editor.addEventListener("input", () => {
                    localStorage.setItem(storageKey(TEXT_KEY_PREFIX, item.relativeImagePath), editor.value);
                    resizeEditor(editor);
                });

                rotationSelect.addEventListener("change", () => {
                    const selectedRotation = parseInt(rotationSelect.value, 10) || 0;
                    localStorage.setItem(storageKey(DISPLAY_ROTATION_KEY_PREFIX, item.relativeImagePath), String(selectedRotation));
                    applyImageTransform(image, selectedRotation, entryRef.zoomLevel);
                });

                function updateZoom(nextZoom) {
                    entryRef.zoomLevel = Math.min(2.5, Math.max(0.5, nextZoom));
                    localStorage.setItem(zoomKey, String(entryRef.zoomLevel));
                    const selectedRotation = parseInt(rotationSelect.value, 10) || 0;
                    applyImageTransform(image, selectedRotation, entryRef.zoomLevel);
                }

                zoomOutButton.addEventListener("click", () => {
                    updateZoom(entryRef.zoomLevel - 0.1);
                });

                zoomInButton.addEventListener("click", () => {
                    updateZoom(entryRef.zoomLevel + 0.1);
                });

                completeCheckbox.addEventListener("change", () => {
                    updateCompleteState(entryRef);
                });

                playButton.addEventListener("click", () => {
                    if (audio.paused) {
                        audio.currentTime = 0;
                        audio.play();
                    } else {
                        audio.pause();
                    }
                });

                audio.addEventListener("play", () => updatePlayButtonLabel(audio, playButton));
                audio.addEventListener("pause", () => updatePlayButtonLabel(audio, playButton));
                audio.addEventListener("ended", () => updatePlayButtonLabel(audio, playButton));

                let mediaRecorder = null;
                let chunks = [];

                recordButton.addEventListener("click", async () => {
                    if (mediaRecorder && mediaRecorder.state === "recording") {
                        mediaRecorder.stop();
                        return;
                    }
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        chunks = [];
                        mediaRecorder = new MediaRecorder(stream, { mimeType: RECORDING_TYPE });

                        mediaRecorder.addEventListener("dataavailable", (event) => {
                            if (event.data && event.data.size > 0) {
                                chunks.push(event.data);
                            }
                        });

                        mediaRecorder.addEventListener("stop", () => {
                            const blob = new Blob(chunks, { type: RECORDING_TYPE });
                            const reader = new FileReader();
                            reader.onloadend = () => {
                                const dataUrl = reader.result;
                                localStorage.setItem(storageKey(AUDIO_KEY_PREFIX, item.relativeImagePath), dataUrl);
                                audio.src = dataUrl;
                                playButton.hidden = false;
                                status.textContent = "Recording saved";
                                updatePlayButtonLabel(audio, playButton);
                            };
                            reader.readAsDataURL(blob);
                            stream.getTracks().forEach((track) => track.stop());
                            recordButton.classList.remove("recording");
                            recordButton.textContent = "Record";
                        });

                        mediaRecorder.start();
                        recordButton.classList.add("recording");
                        recordButton.textContent = "Stop";
                        status.textContent = "Recording...";
                    } catch (error) {
                        status.textContent = "Microphone unavailable";
                    }
                });

                updateCompleteState(entryRef, true);
                entry.classList.toggle("hidden-by-toggle", !isShowingComplete() && completeCheckbox.checked);
                return entryRef;
            }

            function renderPage() {
                const grouped = new Map();
                pageData.forEach((item) => {
                    if (!grouped.has(item.groupId)) {
                        grouped.set(item.groupId, {
                            groupId: item.groupId,
                            groupLabel: item.groupLabel,
                            items: [],
                        });
                        groupOrder.push(item.groupId);
                    }
                    grouped.get(item.groupId).items.push(item);
                });
                const orderedGroupIds = Array.from(grouped.keys()).sort((left, right) => {
                    if (left === "group-none") {
                        return -1;
                    }
                    if (right === "group-none") {
                        return 1;
                    }
                    return grouped.get(left).groupLabel.localeCompare(grouped.get(right).groupLabel);
                });

                if (pageData.length === 0) {
                    const empty = document.createElement("div");
                    empty.className = "empty-state";
                    empty.textContent = "No images were found.";
                    report.appendChild(empty);
                    return;
                }

                groupNav.appendChild(new Option("Jump to subfolder", "", true, true));

                orderedGroupIds.forEach((groupId) => {
                    const group = grouped.get(groupId);
                    const section = document.createElement("section");
                    section.className = "group-section";
                    section.id = groupId;

                    const header = document.createElement("div");
                    header.className = "group-header";
                    header.textContent = group.groupLabel === "none" ? "none" : group.groupLabel;

                    const items = document.createElement("div");
                    items.className = "group-items";

                    group.items.forEach((item) => {
                        const entry = buildEntry(item);
                        items.appendChild(entry.element);
                    });

                    section.appendChild(header);
                    section.appendChild(items);
                    report.appendChild(section);
                    groupNav.appendChild(new Option(group.groupLabel, groupId));
                });
            }

            renderPage();
            applyCompleteVisibility();

            headerThumb.src = "andrew.png";

            requestAnimationFrame(() => {
                document.querySelectorAll("[data-text-editor]").forEach(resizeEditor);
            });
            window.addEventListener("load", () => {
                document.querySelectorAll("[data-text-editor]").forEach(resizeEditor);
            });

            groupNav.addEventListener("change", () => {
                if (!groupNav.value) {
                    return;
                }
                const target = document.getElementById(groupNav.value);
                if (target) {
                    target.scrollIntoView({ behavior: "smooth", block: "start" });
                }
            });

            toggleCompleteButton.addEventListener("click", () => {
                localStorage.setItem(SHOW_COMPLETE_KEY, isShowingComplete() ? "false" : "true");
                applyCompleteVisibility();
            });

            window.addEventListener("scroll", () => {
                scrollTopButton.classList.toggle("visible", window.scrollY > 240);
            });

            scrollTopButton.addEventListener("click", () => {
                window.scrollTo({ top: 0, behavior: "smooth" });
            });

            downloadButton.addEventListener("click", async () => {
                const zip = new JSZip();
                document.querySelectorAll(".image-entry").forEach((entry) => {
                    const fileKey = entry.dataset.fileKey;
                    const editor = entry.querySelector("[data-text-editor]");
                    const recordingDataUrl = localStorage.getItem(storageKey(AUDIO_KEY_PREFIX, fileKey));
                    zip.file(toZipPath("texts", fileKey, ".txt"), editor.value);
                    if (recordingDataUrl) {
                        zip.file(toZipPath("recordings", fileKey, ".webm"), dataUrlToBlob(recordingDataUrl));
                    }
                });

                const content = await zip.generateAsync({ type: "blob" });
                const url = URL.createObjectURL(content);
                const link = document.createElement("a");
                link.href = url;
                link.download = "andrews-meanderings-export.zip";
                document.body.appendChild(link);
                link.click();
                link.remove();
                URL.revokeObjectURL(url);
            });
        })();
    </script>
</body>
</html>
""".replace("__PAGE_DATA__", page_data)

    return html_content


def generate_html_report(processed_data):
    html_content = build_html_page(processed_data)
    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated HTML report at: {HTML_OUTPUT.resolve()}")

def main():
    image_files = iter_image_files()
    if not image_files:
        print(f"No valid images found in '{IMAGE_FOLDER}'.")
        return
    print(f"Found {len(image_files)} images. Starting pipeline...")
    results_for_html = []
    master_keywords_dict = {}
    for img_path in tqdm(image_files, desc="Processing Items"):
        fix_image_rotation(img_path)
        relative_image_path, group_key, group_label, group_id = relative_image_info(img_path)
        txt_path = mirrored_text_path(relative_image_path)
        json_path = mirrored_keyword_path(relative_image_path)
        orientation_path = mirrored_orientation_path(relative_image_path)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        orientation_path.parent.mkdir(parents=True, exist_ok=True)

        if orientation_path.exists():
            try:
                with open(orientation_path, "r", encoding="utf-8") as f:
                    cached_orientation = json.load(f)
                display_rotation = int(cached_orientation.get("rotation", 0))
                cached_version = int(cached_orientation.get("modelVersion", 0))
                if display_rotation not in (0, 90, 180, 270):
                    display_rotation = 0
                if cached_version != ORIENTATION_MODEL_VERSION:
                    display_rotation = detect_display_rotation(img_path)
            except Exception:
                display_rotation = detect_display_rotation(img_path)
        else:
            display_rotation = detect_display_rotation(img_path)

        with open(orientation_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rotation": display_rotation,
                    "modelVersion": ORIENTATION_MODEL_VERSION,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

        aligned_image_path = create_aligned_display_image(img_path, relative_image_path, display_rotation)
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                transcription = f.read()
        else:
            transcription = perform_ocr(img_path)
            if not transcription.startswith("ERROR processing image:"):
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(transcription)
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    keywords = json.load(f)
                except json.JSONDecodeError:
                    keywords = extract_keywords(transcription)
        else:
            if transcription.startswith("ERROR processing image:"):
                keywords = ["OCR Error"]
            else:
                keywords = extract_keywords(transcription)
            if keywords != ["Error extracting keywords"]:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(keywords, f, ensure_ascii=False, indent=4)
        results_for_html.append(
            {
                "relativeImagePath": relative_image_path.as_posix(),
                "displayImagePath": os.path.relpath(aligned_image_path, start=HTML_OUTPUT.parent),
                "filename": img_path.name,
                "groupKey": group_key,
                "groupLabel": group_label,
                "groupId": group_id,
                "rotation": display_rotation,
                "text": transcription,
                "keywords": keywords,
                "modifiedTime": os.path.getmtime(aligned_image_path),
            }
        )
        master_keywords_dict[relative_image_path.as_posix()] = keywords
    with open(MASTER_KEYWORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(master_keywords_dict, f, ensure_ascii=False, indent=4)
    print(f"\nSuccessfully compiled master keywords file at: {MASTER_KEYWORDS_FILE.resolve()}")
    generate_html_report(results_for_html)

if __name__ == "__main__":
    main()
