import json
import os
import csv
import asyncio
import base64
import io
import tempfile
import time
import re
import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from langsmith import traceable
import google.generativeai as genai
from google.generativeai.types import file_types
from google import genai as google_genai
from google.genai import types
from PIL import Image

# PDF to image conversion
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
    print("[OK] pdf2image available")
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("[WARNING] pdf2image not installed. Run: pip install pdf2image")

try:
    from openpyxl import Workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Load environment variables
load_dotenv()

# Set up LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "hansard-sequential-classifier"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize Google genai client for classification
try:
    client = google_genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    print("[OK] Google genai client initialized")
except Exception as e:
    print(f"[WARNING] Google genai client not available: {e}")
    client = None

# Initialize Gemini clients for OCR/extraction and translation
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    # Extraction uses google_genai.Client (new API) with code_execution tool - see client above
    gemini_extraction_model = None  # Not used; extraction goes through client directly
    print("[OK] Gemini extraction uses client (gemini-3-flash-preview + code_execution)")
    
    # Gemini 2.5 Flash for translation - faster model
    gemini_translation_model = genai.GenerativeModel("gemini-3-flash-preview")
    print("[OK] Gemini translation model initialized (gemini-3-flash-preview)")
    
except Exception as e:
    print(f"[WARNING] Gemini not available: {e}")
    gemini_extraction_model = None
    gemini_translation_model = None

# ============================================================================
# COST TRACKING
# ============================================================================

MODEL_PRICING = {
    "gemini-3-flash-preview": {
        "input":  0.50,
        "output": 3.00,
    },
    "gemini-2.5-pro": {
        "input_le_200k":  1.25,
        "input_gt_200k":  2.50,
        "output_le_200k": 10.00,
        "output_gt_200k": 15.00,
    },
}
DEFAULT_MODEL = "gemini-3-flash-preview"

cost_tracker: Dict[str, Dict[str, int]] = {}

def _get_model_cost(model: str, inp: int, out: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    if "input" in pricing:
        return inp / 1_000_000 * pricing["input"] + out / 1_000_000 * pricing["output"]
    threshold = 200_000
    inp_rate = pricing["input_le_200k"] if inp <= threshold else pricing["input_gt_200k"]
    out_rate = pricing["output_le_200k"] if inp <= threshold else pricing["output_gt_200k"]
    return inp / 1_000_000 * inp_rate + out / 1_000_000 * out_rate

def _track_cost(response, label: str = "", model: str = DEFAULT_MODEL):
    """Extract usage metadata from a Gemini response and accumulate costs."""
    try:
        usage = response.usage_metadata
        if usage is None:
            return
        inp = getattr(usage, 'prompt_token_count', 0) or 0
        out = getattr(usage, 'candidates_token_count', 0) or 0
        if model not in cost_tracker:
            cost_tracker[model] = {"input_tokens": 0, "output_tokens": 0}
        cost_tracker[model]["input_tokens"]  += inp
        cost_tracker[model]["output_tokens"] += out
        cost = _get_model_cost(model, inp, out)
        tag = f" [{label}]" if label else ""
        print(f"    tokens: {inp} in / {out} out | cost: ${cost:.5f}{tag} ({model})")
    except Exception:
        pass

def print_cost_summary():
    """Print cumulative cost summary."""
    print(f"\n========== COST SUMMARY ==========")
    grand_total = 0.0
    for model, counts in cost_tracker.items():
        inp = counts["input_tokens"]
        out = counts["output_tokens"]
        model_cost = _get_model_cost(model, inp, out)
        grand_total += model_cost
        print(f"  [{model}]")
        print(f"    Input tokens  : {inp:,}")
        print(f"    Output tokens : {out:,}")
        print(f"    Cost          : ${model_cost:.5f}")
    print(f"  --------------------------------")
    print(f"  Total cost      : ${grand_total:.5f}")
    print(f"===================================")

# ============================================================================
# LANGSMITH WRAPPERS FOR GEMINI API CALLS
# ============================================================================

@traceable(name="gemini_generate_content_new_api")
def gemini_generate_content_wrapped(client_instance: google_genai.Client, model: str, contents: str, config: types.GenerateContentConfig = None):
    """
    LangSmith-wrapped function for Google genai Client.generate_content calls
    """
    response = client_instance.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )
    _track_cost(response, "new_api", model=model)
    return response

@traceable(name="gemini_generate_content_old_api")
def gemini_generate_content_old_wrapped(model_instance, prompt, image=None, generation_config=None, model_name: str = DEFAULT_MODEL):
    """
    LangSmith-wrapped function for old genai.GenerativeModel.generate_content calls
    """
    if image is not None:
        response = model_instance.generate_content([prompt, image])
    else:
        if generation_config is not None:
            response = model_instance.generate_content(prompt, generation_config=generation_config)
        else:
            response = model_instance.generate_content(prompt)
    _track_cost(response, "old_api", model=model_name)
    return response


# ============================================================================
# PDF TO IMAGE CONVERSION UTILITIES
# ============================================================================

def convert_pdf_to_images(pdf_path: str, dpi: int = 200, output_dir: str = None) -> List[Dict[str, Any]]:
    """
    Convert a PDF file to a list of page images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for conversion (default 200 for good OCR quality)
        output_dir: Optional directory to save images (if None, keeps in memory)
    
    Returns:
        List of dicts with page info and image data
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image is not installed. Run: pip install pdf2image")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Converting PDF to images: {pdf_path}")
    print(f"  DPI: {dpi}")
    
    # Convert PDF to images
    images = convert_from_path(str(pdf_path), dpi=dpi)
    
    print(f"  Total pages: {len(images)}")
    
    page_data_list = []
    
    for page_num, image in enumerate(images, 1):
        page_data = {
            "page": page_num,
            "image": image,  # PIL Image object
            "width": image.width,
            "height": image.height,
            "metadata": {
                "source": str(pdf_path),
                "page": page_num,
                "total_pages": len(images),
                "dpi": dpi
            }
        }
        
        # Optionally save to disk
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            image_path = output_path / f"page_{page_num:04d}.png"
            image.save(str(image_path), "PNG")
            page_data["image_path"] = str(image_path)
        
        page_data_list.append(page_data)
    
    print(f"  Conversion complete!")
    return page_data_list


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_image_from_path(image_path: str) -> Image.Image:
    """Load image from file path."""
    return Image.open(image_path)


def resize_image_for_api(image: Image.Image, max_size: int = 2048) -> Image.Image:
    """
    Resize image to fit within max_size while maintaining aspect ratio.
    This prevents Gemini API timeouts for large images.
    """
    width, height = image.size
    
    # If already small enough, return as-is
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate new size maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Use high-quality downsampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized
# Pydantic models for data validation
class StructureFlags(BaseModel):
    has_tables: bool = False
    heading_text: List[str] = []

    @field_validator('heading_text', mode='before')
    @classmethod
    def _coerce_heading_text(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith('[') and v.endswith(']'):
                try:
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, list) else [str(parsed)]
                except (json.JSONDecodeError, ValueError):
                    return [v.strip('[]').strip()]
            return [v]
        return [str(v)]

class Theme(BaseModel):
    main_category: str
    subcategory: str
    justification: str

class ClassifiedSpeech(BaseModel):
    page: int
    speaker_name: Optional[str] = ""
    speech: str
    theme: Theme
    structure_flags: StructureFlags
    input_id: Union[int, str]
    input_type: str = ""
    original_speech: str = ""  # Original text before translation
    speech_start_time: str = ""  # Hansard clock line in brackets above the speaker, e.g. [පූ.භා. 11.12]
    speech_start_time_english: str = ""  # From translation agent: time in plain English for input_id rules
    has_petition: str = "no"  # "yes" if input_id is 11, "no" otherwise
    petition_count: int = 0  # Number of petitions submitted within the speech (only relevant for input_id 11)
    has_expunged: str = "no"  # "yes" if ANY part of the speech contains expunged content, "no" otherwise
    expunged_count: int = 0  # Number of expunged statements (from LLM per speech; summed when merging)

    @model_validator(mode='after')
    def set_flags_from_input_id(self):
        """Automatically set has_petition based on input_id. has_expunged is set by LLM based on content."""
        if self.input_id == 11:
            self.has_petition = "yes"
        # Note: has_expunged should be set by LLM based on content detection, not just input_id
        # Only set if input_id is 13 (entire speech expunged) as a fallback
        if self.input_id == 13 and self.has_expunged == "no":
            self.has_expunged = "yes"
        if self.has_expunged == "no":
            self.expunged_count = 0
        elif self.expunged_count < 1:
            self.expunged_count = 1
        return self

class RawSpeech(BaseModel):
    """Model for raw speeches extracted from pages"""
    speaker: Optional[str] = None
    speech: str
    heading: Optional[str] = None
    speech_start_time: Optional[str] = None  # Time line in square brackets above trilingual name

class ExtractedSpeech(BaseModel):
    """Model for extracted speeches after translation and structure analysis"""
    speaker_name: Optional[str] = ""
    speech_text: Optional[str] = ""
    speech: Optional[str] = ""
    heading_text: List[str] = []
    original_speech: str = ""
    speech_start_time: str = ""
    speech_start_time_english: str = ""  # Translated/normalized clock time for classifiers (e.g. "11:12 a.m.")
    structure_flags: Optional[StructureFlags] = None

    @field_validator('heading_text', mode='before')
    @classmethod
    def _coerce_heading_text(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith('[') and v.endswith(']'):
                try:
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, list) else [str(parsed)]
                except (json.JSONDecodeError, ValueError):
                    return [v.strip('[]').strip()]
            return [v]
        return [str(v)]

class PageData(BaseModel):
    """Model for page data"""
    page: Union[int, str] = "unknown"
    original_text: str = ""
    translated_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    physical_pages: List[int] = Field(default_factory=list)
    image: Optional[Any] = None  # PIL Image or image path
    image_path: Optional[str] = None

# Input ID mapping for classification types
INPUT_ID_MAPPING = {
    1: "Written Question",
    2: "Written Question - Response", 
    3: "Written Question - Follow up question",
    4: "Follow up question - Response",
    5: "Written Responses",
    6: "Bill / Regulation / Order /Resolution - Administration",
    7: "Bill / Regulation / Order /Resolution - Debate Oral Contribution",
    8: "Point of Order- Technical/Procedural",
    9: "Point of Order - Other",
    10: "Motion at the Adjournment Time",
    11: "Petitions",
    13: "Expunged Statement",
    14: "Oral Contribution",
    15: "Oral Contribution - Core Statements",
    16: "privileges",
    17: "Question by private notice",
    18: "Question by private notice - Ans",
    19: "PMQ",
    "19-1": "PMQ - Follow-Up Question (1st)",
    "19-2": "PMQ - Follow-Up Question (2nd)",
    20: "Notification",
    21: "Announcement",
    22: "PMM",
    23: "PMQ-Responses",
    "23-1": "PMQ-Responses - Follow-Up Answer (1st)",
    "23-2": "PMQ-Responses - Follow-Up Answer (2nd)",
    24: "Adjournment Question",
    25: "Adjournment Question-Responses",
    26: "Private Member's Bill - Public interest",
    27: "Budget Speech",
    28: "Adjournment Motion",
    29: "President's Speech",
    30: "Address"
}

# Input ID scoring rules
FIXED_INPUT_ID_SCORES = {
    1: 30,
    2: 30,
    3: 10,
    4: 10,
    5: 30,
    8: 30,
    9: -30,
    11: 20,
    17: 30,
    18: 30,
    19: 30,
    "19-1": 10,
    "19-2": 10,
    22: 180,
    23: 30,
    "23-1": 10,
    "23-2": 10,
    24: 30,
    25: 30,
    26: 240,
    27: 400,
}

LINE_WEIGHT_INPUT_ID_SCORES = {
    6: 0.05,
    7: 0.5,
    10: 1,
    12: -1,
    14: 0.05,
    15: 0.5,
    16: 0.05,
    20: 0.05,
    21: 0.05,
    28: 0.5,
}


def calculate_input_score(input_id: Union[int, str], line_count: int, petition_count: int = 0) -> float:
    """
    Calculate score from input_id, line count, and petition count.
    Fixed-score IDs use direct mapping; weighted IDs use line_count * weight.
    Special case: input_id 11 uses petition_count * 20.
    """
    normalized_input_id: Union[int, str] = input_id
    if isinstance(input_id, str):
        normalized_input_id = input_id.strip()
        if normalized_input_id.isdigit():
            normalized_input_id = int(normalized_input_id)

    if normalized_input_id == 11:
        petitions = max(int(petition_count or 0), 0)
        return float(petitions * 20)

    if normalized_input_id in FIXED_INPUT_ID_SCORES:
        return float(FIXED_INPUT_ID_SCORES[normalized_input_id])

    if normalized_input_id in LINE_WEIGHT_INPUT_ID_SCORES:
        return float(line_count * LINE_WEIGHT_INPUT_ID_SCORES[normalized_input_id])

    return 0.0


# ============================================================================
# SHORT SPEECH FILTERING (post-processing)
# ============================================================================

# Keep these input IDs even if computed line count is < 5
KEEP_SHORT_SPEECH_INPUT_IDS = {
    1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 30
}


def base_input_id(value: Any) -> Optional[int]:
    """
    Normalize input_id to a base integer if possible.
    Examples: 19 -> 19, "19" -> 19, "19-1" -> 19, " 23-2 " -> 23.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.match(r"^\s*(\d+)", value)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def speech_line_count_word_based(speech_dict: Dict[str, Any]) -> int:
    """
    Word-based line count consistent with CSV export:
      line_count = ceil(words / 6.3)
    Uses original_speech if present, else falls back to speech.
    """
    text = (speech_dict.get("original_speech") or speech_dict.get("speech") or "").strip()
    if not text:
        return 0
    words = len(text.split())
    if words <= 0:
        return 0
    return int(math.ceil(words / 6.3))


def filter_short_speeches(
    speeches: List[Dict[str, Any]],
    min_lines: int = 5,
    keep_input_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if not speeches:
        return speeches

    keep_input_ids = keep_input_ids or KEEP_SHORT_SPEECH_INPUT_IDS

    kept: List[Dict[str, Any]] = []
    removed = 0
    removed_by_base_id: Dict[Union[int, str], int] = {}

    for s in speeches:
        line_count = speech_line_count_word_based(s)
        bid = base_input_id(s.get("input_id"))

        if line_count >= min_lines or (bid is not None and bid in keep_input_ids):
            kept.append(s)
        else:
            removed += 1
            key: Union[int, str] = bid if bid is not None else str(s.get("input_id", "Unknown"))
            removed_by_base_id[key] = removed_by_base_id.get(key, 0) + 1

    if removed:
        top_removed = sorted(removed_by_base_id.items(), key=lambda x: x[1], reverse=True)[:8]
        top_str = ", ".join([f"{k}:{v}" for k, v in top_removed])
        print(f"  [OK] Short-speech filter: kept={len(kept)} removed={removed} (top removed base_ids: {top_str})")
    else:
        print(f"  [OK] Short-speech filter: kept={len(kept)} removed=0")

    return kept


def merge_similar_speaker_speeches(speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    After short-speech filtering, merge consecutive speeches whose speaker
    names are similar back into a single entry.

    For merged groups:
      - theme, input_id, input_type come from the segment with the most words
      - speech / original_speech texts are concatenated in order
      - has_tables is True if ANY segment had it
      - has_expunged is "yes" if ANY segment had it
      - expunged_count is the SUM of each segment's expunged_count
      - has_petition / petition_count are preserved from any segment that had them

    AUDITOR-GENERAL'S REPORT / PAPERS PRESENTED: no merge when both segments
    share the same page; cross-page adjacency still merges.
    """
    if not speeches or len(speeches) <= 1:
        return speeches

    def _extract_key_words(name: str) -> List[str]:
        titles = ['the hon.', 'hon.', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'minister', 'deputy', 'speaker']
        words = name.split()
        key_words = [w for w in words if not any(t in w.lower() for t in titles)]
        return [w.lower().strip('.,()') for w in key_words if len(w) > 2]

    def _similar(n1: str, n2: str) -> bool:
        if not n1 or not n2:
            return False
        a, b = n1.strip().lower(), n2.strip().lower()
        if a == b:
            return True
        w1, w2 = _extract_key_words(n1), _extract_key_words(n2)
        if not w1 or not w2:
            return False
        if len(set(w1) & set(w2)) >= min(2, min(len(w1), len(w2))):
            return True
        if len(a) > 5 and len(b) > 5:
            if a in b or b in a:
                return True
            if abs(len(a) - len(b)) <= 2:
                c1, c2 = set(a.replace(' ', '')), set(b.replace(' ', ''))
                if (len(c1 & c2) / max(len(c1), len(c2), 1)) > 0.7:
                    return True
        return False

    def _word_count(s: Dict[str, Any]) -> int:
        txt = (s.get('original_speech') or s.get('speech') or '').strip()
        return len(txt.split()) if txt else 0

    def _merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        for s in group:
            s['_wc'] = _word_count(s)
        best = max(group, key=lambda s: s['_wc'])
        merged = best.copy()

        all_speech = [s.get('speech', '') for s in group]
        merged['speech'] = ' '.join(filter(None, all_speech))

        all_orig = [s.get('original_speech', '') for s in group if s.get('original_speech')]
        if all_orig:
            merged['original_speech'] = ' '.join(all_orig)

        total_words = sum(s['_wc'] for s in group)
        merged['word_count'] = total_words
        merged['aggregated_word_count'] = total_words
        merged['original_speech_word_count'] = len(merged.get('original_speech', '').split())

        pages = sorted({s.get('page', 0) for s in group})
        merged['pages'] = pages
        merged['page'] = pages[0]

        merged['speaker_name'] = best.get('speaker_name', '')
        merged['is_aggregated'] = True
        merged['aggregated_count'] = len(group)

        if any(s.get('structure_flags', {}).get('has_tables', False) for s in group):
            sf = merged.get('structure_flags', {})
            if isinstance(sf, dict):
                sf['has_tables'] = True
            elif hasattr(sf, 'has_tables'):
                sf.has_tables = True
            merged['structure_flags'] = sf

        if any(s.get('has_expunged') == 'yes' for s in group):
            merged['has_expunged'] = 'yes'
        merged['expunged_count'] = sum(int(s.get('expunged_count', 0) or 0) for s in group)

        merged['speech_start_time'] = ''
        merged['speech_start_time_english'] = ''
        for s in group:
            t = (s.get('speech_start_time') or '').strip()
            if t:
                merged['speech_start_time'] = t
                break
        for s in group:
            te = (s.get('speech_start_time_english') or '').strip()
            if te:
                merged['speech_start_time_english'] = te
                break

        if any(s.get('has_petition') == 'yes' for s in group):
            merged['has_petition'] = 'yes'
            merged['petition_count'] = sum(s.get('petition_count', 0) for s in group)

        for s in group:
            s.pop('_wc', None)
        merged.pop('_wc', None)

        return merged

    NO_MERGE_HEADINGS = {"AUDITOR GENERAL'S REPORT", "AUDITOR-GENERAL'S REPORT", "PAPERS PRESENTED"}

    def _heading_blocked(s: Dict[str, Any]) -> bool:
        for src in [s.get('structure_flags', {}).get('heading_text', []), s.get('heading_text', [])]:
            if not src:
                continue
            for h in (src if isinstance(src, list) else [src]):
                if str(h).strip().upper() in NO_MERGE_HEADINGS:
                    return True
        return False

    result: List[Dict[str, Any]] = []
    group: List[Dict[str, Any]] = [speeches[0]]

    for s in speeches[1:]:
        prev_speaker = group[-1].get('speaker_name', '').strip()
        curr_speaker = s.get('speaker_name', '').strip()
        prev_p = group[-1].get('page', 0)
        curr_p = s.get('page', 0)
        blocked = _heading_blocked(group[-1]) or _heading_blocked(s)
        blocked_same_page = blocked and (prev_p == curr_p)
        if _similar(prev_speaker, curr_speaker) and not blocked_same_page:
            group.append(s)
        else:
            result.append(_merge_group(group) if len(group) > 1 else group[0])
            group = [s]

    result.append(_merge_group(group) if len(group) > 1 else group[0])

    merged_count = len(speeches) - len(result)
    print(f"  [OK] Similar-speaker merge: {len(speeches)} -> {len(result)} speeches ({merged_count} merged back)")
    return result


QUESTION_RESPONSE_MAP: Dict[Union[int, str], set] = {
    1: {2},
    3: {4},
    17: {18},
    19: {23},
    "19-1": {"23-1"},
    "19-2": {"23-2"},
    24: {25},
}

SKIP_IDS_FOR_ANSWER_SCAN = {8, 9, 14, 30}


def mark_answered_questions(speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each speech whose input_id is a question type, scan forward within the
    same heading section to determine if a matching response follows.
    Sets 'was_answered' to 'yes' / 'no' on question speeches only.
    """
    answered_count = 0
    unanswered_count = 0

    for i, speech in enumerate(speeches):
        qid = speech.get('input_id')
        norm_qid = qid
        if isinstance(qid, str):
            norm_qid = qid.strip()
            if norm_qid.isdigit():
                norm_qid = int(norm_qid)

        if norm_qid not in QUESTION_RESPONSE_MAP:
            continue

        expected = QUESTION_RESPONSE_MAP[norm_qid]
        heading = speech.get('structure_flags', {}).get('heading_text', [])
        answered = False

        for j in range(i + 1, len(speeches)):
            ns = speeches[j]
            nid = ns.get('input_id')
            norm_nid = nid
            if isinstance(nid, str):
                norm_nid = nid.strip()
                if norm_nid.isdigit():
                    norm_nid = int(norm_nid)

            next_heading = ns.get('structure_flags', {}).get('heading_text', [])
            if next_heading != heading:
                break

            if norm_nid in expected:
                answered = True
                break

            bid = base_input_id(nid)
            if bid in SKIP_IDS_FOR_ANSWER_SCAN:
                continue

            break

        speech['was_answered'] = 'yes' if answered else 'no'
        if answered:
            answered_count += 1
        else:
            unanswered_count += 1

    print(f"  [OK] Questions answered: {answered_count}, unanswered: {unanswered_count}")
    return speeches


def remove_address_rows(speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove speeches with input_id 30 (Address) from the final output."""
    before = len(speeches)
    filtered = [s for s in speeches if base_input_id(s.get('input_id')) != 30]
    removed = before - len(filtered)
    print(f"  [OK] Removed {removed} address row(s) (input_id 30)")
    return filtered


# Main Category Mapping
MAIN_CATEGORY_MAPPING = {
    1: "Agriculture, Plantations, Livestock & Fisheries",
    2: "Natural Resources & Environment",
    3: "Reconciliation & Resettlement",
    4: "Trade & Industry",
    5: "Welfare & Social Services",
    6: "Justice, Defense & Public Order",
    7: "National Heritage, Media & Sports",
    8: "Economy and Finance",
    9: "Education",
    10: "Labour & Employment",
    11: "Technology, Communications & Energy",
    12: "Governance, Administration and Parliamentary Affairs",
    13: "Health",
    14: "Urban Planning, Infrastructure and Transportation",
    15: "Rights & Representation"
}

# Sub Category Mapping
SUB_CATEGORY_MAPPING = {
    '1a': "Livestock & Dairy",
    '1b': "Food and Nutrition",
    '1c': "Plantations & minor Export Crops",
    '1d': "Fisheries and Aquatic Resources",
    '1e': "Paddy Cultivation & Traditional Agriculture",
    '2a': "Forestry, Wild Life and Natural Resources",
    '2b': "Land",
    '2c': "Disaster Management",
    '2d': "Environment and Sustainable Development",
    '3a': "Reconciliation",
    '3b': "Resettlement",
    '4a': "International Trade",
    '4b': "Domestic Trade & Industry",
    '4c': "Public Enterprises",
    '4d': "Tourism",
    '4e': "Consumer Affairs",
    '5a': "Welfare",
    '5b': "Social Services",
    '5c': "Vulnerable Groups",
    '6a': "Law & Order",
    '6b': "Defense",
    '6c': "Prisons & Rehabilitation",
    '6d': "Justice",
    '7a': "Art & Culture",
    '7b': "Religion",
    '7c': "Media",
    '7d': "Sports",
    '8a': "Taxes and Other Government Revenue",
    '8b': "Government Expenditure",
    '8c': "Debt",
    '8d': "Economic Policy and Development",
    '9a': "Schools",
    '9b': "University",
    '9c': "Vocational",
    '10a': "Salaries & Social Security",
    '10b': "Foreign Employment",
    '10c': "Jobs and unemployment",
    '10d': "Labour Rights and Trade Unions",
    '11a': "Electricity",
    '11b': "Petroleum & Gas",
    '11c': "Science and Technology",
    '11d': "Telecommunications and Information Technology",
    '12a': "Governance/Constitutional Reforms",
    '12b': "Foreign Affairs",
    '12c': "Public Administration",
    '12d': "Provincial Councils and Local Government",
    '12e': "Parliamentary Affairs",
    '12f': "Parliamentary Appreciations",
    '12g': "Good Governance",
    '13a': "Health Services",
    '13b': "Diseases",
    '13c': "Medicine",
    '13d': "Traditional Medicine",
    '14a': "Highways",
    '14b': "Ports, Shipping and Aviation",
    '14c': "Public Transport",
    '14d': "Private Transport",
    '14e': "Urban Development",
    '14f': "Water Supply and Drainage",
    '14g': "Housing & construction",
    '15a': "Women",
    '15b': "Youth",
    '15c': "Minorities",
    '15d': "Human Rights",
    '15e': "Children"
}

TAXONOMY = """
1. Agriculture, Plantations, Livestock & Fisheries
   a. Livestock & Dairy
   b. Food and Nutrition
   c. Plantations & minor Export Crops
   d. Fisheries and Aquatic Resources
   e. Paddy Cultivation & Traditional Agriculture

2. Natural Resources & Environment
   a. Forestry, Wild Life and Natural Resources
   b. Land
   c. Disaster Management
   d. Environment and Sustainable Development

3. Reconciliation & Resettlement
   a. Reconciliation
   b. Resettlement

4. Trade & Industry
   a. International Trade
   b. Domestic Trade & Industry
   c. Public Enterprises
   d. Tourism
   e. Consumer Affairs

5. Welfare & Social Services
   a. Welfare
   b. Social Services
   c. Vulnerable Groups

6. Justice, Defense & Public Order
   a. Law & Order
   b. Defense
   c. Prisons & Rehabilitation
   d. Justice

7. National Heritage, Media & Sports
   a. Art & Culture
   b. Religion
   c. Media
   d. Sports

8. Economy and Finance
   a. Taxes and Other Government Revenue
   b. Government Expenditure
   c. Debt
   d. Economic Policy and Development

9. Education
   a. Schools
   b. University
   c. Vocational

10. Labour & Employment
    a. Salaries & Social Security
    b. Foreign Employment
    c. Jobs and unemployment
    d. Labour Rights and Trade Unions

11. Technology, Communications & Energy
    a. Electricity
    b. Petroleum & Gas
    c. Science and Technology
    d. Telecommunications and Information Technology

12. Governance, Administration and Parliamentary Affairs
    a. Governance/Constitutional Reforms
    b. Foreign Affairs
    c. Public Administration
    d. Provincial Councils and Local Government
    e. Parliamentary Affairs
    f. Parliamentary Appreciations
    g. Good Governance

13. Health
    a. Health Services
    b. Diseases
    c. Medicine
    d. Traditional Medicine

14. Urban Planning, Infrastructure and Transportation
    a. Highways
    b. Ports, Shipping and Aviation
    c. Public Transport
    d. Private Transport
    e. Urban Development
    f. Water Supply and Drainage
    g. Housing & Construction

15. Rights & Representation
    a. Women
    b. Youth
    c. Minorities
    d. Human Rights
    e. Children
"""

# Async Agent 1: Gemini-based Speaker Extraction Agent (Async Version with Image Support)
@traceable
async def async_speaker_extraction_agent(page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Async version: Extract speakers and speeches from trilingual Hansard using Gemini Vision.
    Supports both text input and image input (for PDF pages).
    Includes retry logic for 504 timeout errors with 2-minute timeout per page.
    """
    # Extract page number from metadata
    page_number = page_data.get('metadata', {}).get('page', page_data.get('page', 'unknown'))
    if page_number != 'unknown':
        try:
            page_number = int(page_number)
        except (ValueError, TypeError):
            page_number = 'unknown'

    # Check if we have an image (PDF page) or text
    has_image = "image" in page_data or "image_path" in page_data
    text = page_data.get("original_text", "") or page_data.get("translated_text", "")
    text = text.strip() if text else ""

    if not has_image and not text:
        return []

    extraction_prompt = f"""You are a specialized data extraction AI focused on processing Parliamentary Hansard documents. Your task is to extract speaker names, section headings, and their corresponding speeches from the provided page image/text containing mixed languages (Sinhala, Sri Lankan Tamil, English).

Output Format: You must return a JSON Array of objects. Each object must follow this schema:

```json
{{
  "speaker": "Name of the Speaker (in English) OR empty string \"\" if not found/visible",
  "speech": "The full text of the speech",
  "heading": "The section heading in English (ALL CAPS) or null if no heading",
  "speech_start_time": "ONLY if a session time appears in square brackets on the line directly above that speaker's trilingual name: copy that full bracketed line EXACTLY (including brackets). If there is NO such time line for this speech, use empty string \"\" — do not guess or invent a time."
}}
```

## Task Instructions:

**⚠️ CRITICAL RULE: EXTRACT EXACT TEXT - DO NOT CHANGE ANYTHING ⚠️**
- Your ONLY task is to COPY the text exactly as written
- Go through LINE BY LINE and extract the SAME content
- DO NOT paraphrase, summarize, shorten, reword, or modify ANY text
- If you change even ONE word, you have FAILED
- IF A BLANK PAGE IS PRESENT, DONT INVENT SPEECHES FROM YOUR OWN. RETURN AN EMPTY STRING.

**DOCUMENT FORMAT**: This is a trilingual Hansard document where:
- **New speakers** are denoted by their name appearing in ALL THREE LANGUAGES in sequence: Sinhala → Tamil (in parentheses) → English (in parentheses)
- Example: "ගරු චමින්ද්‍රනී කිරිඇල්ල" (Sinhala) then "(மாண்புமிகு சமிந்திரானீ)" (Tamil) then "(The Hon. Chamindranee Kiriella)" (English)
- **Continuations** on new pages may show: [Sinhala name only in square brackets] - this means same speaker continuing from previous page
- All three languages appear together for the SAME person when starting a NEW speech block

### Heading Detection (IMPORTANT):
- **Headings are trilingual**: They appear in Sinhala, Sri Lankan Tamil, and English on the same page
- **English headings are in ALL CAPITAL LETTERS** - this is the key identifier
- Extract ONLY the English version of the heading for the "heading" field
- **ALWAYS extract BOTH the main heading AND the sub-heading if both are present**:
  * Main heading example: "QUESTION BY PRIVATE NOTICE"
  * Sub-heading example: "DIFFICULTIES FACED BY SRI LANKAN MIGRANTS IN RENEWING PASSPORTS"
  * If both exist, set heading to: "MAIN HEADING - SUB HEADING" (joined with " - ") 
  * ADJOURNMENTS have a sub heading after the main heading. Extract this sub heading as well.
  * If only one heading level exists, set heading to that heading alone
  * If there is a roman number before the heading, extract that roman number as well.That roman number means the second part under the main heading.For example, QUESTION BY PRIVATE NOTICE - II IMPROVEMENT OF SERVICES PROVIDED BY DEPARTMENT OF PENSIONS 
- Common heading patterns: "NATIONAL BUILDING RESEARCH INSTITUTE BILL", "ORAL ANSWERS TO QUESTIONS", "BILLS PRESENTED", "PETITIONS", "QUESTION BY PRIVATE NOTICE", etc.
- A heading applies to ALL speeches that follow it until a NEW heading appears
- If NO heading is visible above a speech on this page, set heading to null


### Speech start time (IMPORTANT):
- **Not every speech has a time.** Many entries will have `speech_start_time` as `""`.
- **Rule**: If and only if you see the **session clock line** in square brackets **immediately above** the trilingual speaker name block, put that **entire line** in `speech_start_time` — same characters as printed, **do not translate or paraphrase** (e.g. `[පූ.භා. 11.12]`).
- **If there is no such time line above that name**, set `speech_start_time` to `""`. Do **not** infer time from elsewhere on the page; do **not** invent or default a time.
- **Do NOT** put this time line in `speech` — keep `speech` as spoken content only (do not duplicate the time in `speech` when you extracted it to `speech_start_time`).
- **Do NOT** confuse session time with:
  * **Continuation markers** `[ගරු ... මහතා]` (speaker continuation — not a clock line → leave `speech_start_time` as `""` unless a separate clock line exists above the name)
  * Procedural brackets **inside** the speech body (e.g. `[Hon. Members interrupted.]`) — those stay in `speech`, not in `speech_start_time`

### Speaker Identification:
- **NEW SPEECH INDICATOR**: A new speech block is identified when the speaker's name appears in **ALL THREE LANGUAGES** (Sinhala, Sri Lankan Tamil, and English) in sequence
- Example format: "ගරු චමින්ද්‍රනී කිරිඇල්ල නීතිඥ මහත්මිය" (Sinhala) + "(மாண்புமிகு சமிந்திரானீ கிரிஎல்ல)" (Tamil) + "(The Hon. (Mrs.) Chamindranee Kiriella, Attorney-at-Law)" (English)
- **Rule**: Always extract and use the English version of the name for the "speaker" field
- Remove titles like "The Hon.", "Hon.", "Minister of...", "Deputy Minister of...", "Attorney-at-Law", etc.
- For role-based speakers (Speaker, Deputy Speaker, Deputy Chairperson), keep the full role name in English

**CRITICAL SPECIAL CASES FOR PAGE CONTINUATIONS**:
1. **Bracketed Sinhala Name at Page Start** (continuation marker):
   - Format: [ගරු යානමුත්තු ශ්‍රීනේෂන් මහතා] (only in Sinhala, inside square brackets)
   - DO NOT EXTRACT THE SPEAKER NAME FROM THE BRACKETS. return empty string for speaker name for that. 

2. **Page starts with speech text but NO speaker name visible**:
   - Set speaker to "" (empty string). DO NOT guess.
   
3. **Speech appears INCOMPLETE at END of page**:
   - STILL EXTRACT IT - the next page will continue it
   - Look for signs: sentence ends mid-way, paragraph cut off, no concluding punctuation

**Examples**:
- New speech: "ගරු කථානායක" + "தலைவர் அவர்கள்" + "(The Hon. Speaker)" → extract "Speaker"
- Continuation: [ගරු විජේසිරි බස්නායක මහතා] → extract "Wijesiri Basnayake" (from brackets)
- Page starts with speech text but no name → ""

### Speech Extraction & Aggregation:
- Extract the text associated with that speaker.
- **If the same speech content appears in all three languages (Sinhala, Tamil, English), extract ONLY the Sinhala version of the speech text.**
- **IMPORTANT**: If a speaker is identified with a trilingual name block (Sinhala + Tamil + English), extract their speech **EVEN IF IT'S ONLY ONE WORD**
- Examples of short speeches to extract: "Yes", "No", "Thank you", "I agree", "Question", "Point of order"
- If a single speaker's speech spans multiple sections, combine them into a single speech entry.
- Do NOT create separate JSON entries for the same continuous speech by the same person.
- Stop the current speech entry only when a new speaker is explicitly named.
- **EXCEPTION**: Under "PAPERS PRESENTED" or "AUDITOR-GENERAL'S REPORT", create a **separate entry for each individual item** even if the same speaker presents multiple items consecutively. Each paper/report presentation is its own speech entry.

### CRITICAL - Content Preservation (READ CAREFULLY):
- **YOUR ONLY JOB IS TO EXTRACT - DO NOT CHANGE ANYTHING**
- **Go through LINE BY LINE and extract EXACTLY the same content**
- **DO NOT paraphrase, summarize, shorten, or reword ANYTHING**
- **DO NOT skip sentences, clauses, or words**
- **DO NOT change word order or sentence structure**
- **Extract character by character if needed - be that precise**
- **Preserve ALL original text including:**
  * Every word, punctuation mark, and number
  * Interjections like "[Hon. Members interrupted.]"
  * Stage directions in brackets
  * Incomplete sentences
  * Repetitions
  * Everything exactly as it appears
- **If you change even one word, you have FAILED this task**
- Your job is COPYING TEXT, not editing or improving it

### Cleaning & Redaction:
- Remove all metadata tags completely.
- Remove page numbers found at the top or bottom of pages.

### Formatting:
- **Go through LINE BY LINE and extract EXACTLY what you see**
- **Copy the text character by character - do not change anything**
- Extract speech text naturally - preserve how it appears in the document
- Focus on capturing the complete content of each speech
- Do NOT skip any sentences, words, or characters - extract EVERYTHING the speaker said EXACTLY as written

### What to Extract:
- Extract actual speeches, statements, questions, and substantive content.
- **ALWAYS extract ANY speech if the speaker is identified with a trilingual name block** - even single words
- Include interjections, short responses, and meaningful statements.
- Examples: "Yes", "No", "Thank you", "I agree", "Point of order", "Question", actual speeches, etc.
- **Rule**: If you see a trilingual name block (Sinhala + Tamil + English), there MUST be a corresponding speech entry - no matter how short

### BILLS PRESENTED Section (IMPORTANT):
- Under the heading "BILLS PRESENTED", bills are listed with their heading in all three languages.After the English capital letter heading of the particular bill, the actual bill is present in sinhala within double quotes.Extract this as the speech. 
- The presentors name in english is at the end of the section after the sinhala and tamil blocks. Extract this name as the speaker name.The presenter line follows this pattern: "Presented by the Hon. [Name], [Title]; to be read a Second time upon [date]"
- **Extract EACH bill as a separate speech entry**:
  * **speaker**: The presenter's name extracted from the "Presented by..." line (e.g. "Sunil Kumara Gamage")
  * **speech**: The bill presented in sinhala which is within double quotes. Dont extract any other text.
  * **heading**: "BILLS PRESENTED"
- If multiple bills are presented on the same page, create a separate entry for each bill with its respective presenter.
- This is a special case where the bill is recorded before the speaker name. Extract this as well.

### What to EXCLUDE:
- Do NOT extract purely procedural text like "Question put, and agreed to", "Motion carried", etc.

## Output Rules:
- Return ONLY valid JSON array (no explanations, no markdown).
- The response must start with [ and end with ]
- Each entry must have "speaker", "speech", "heading", and "speech_start_time" fields.
- **speech_start_time**: use the bracketed clock line above the name when it exists; otherwise **must** be `""` (empty string). Never invent a time.
- Set "heading" to null if no heading is visible for that speech.
- Some pages are empty. If the page is empty, dont extract anything or come up with invented speeches from your own.If empty return a empty string.
- If speaker name is not found/visible, set "speaker" to "" (empty string). Do NOT use null.
- **REMINDER: The "speech" field must contain EXACT text from the document - no changes allowed**

## Page {page_number}
"""

    # Retry logic: 1 retry attempt after 60 seconds
    retry_attempts = 2  # Initial attempt + 1 retry
    
    for attempt in range(retry_attempts):
        try:
            if client is None:
                raise Exception("Google genai client not initialized")
            
            if has_image:
                # Get image - either PIL Image object or load from path
                if "image" in page_data:
                    image = page_data["image"]
                else:
                    image = Image.open(page_data["image_path"])

                # Convert PIL Image to bytes (same as test_gemini3_ocr.py)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                image_part = types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/png")

                def _extract_sync():
                    return client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=[image_part, extraction_prompt],
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(code_execution=types.ToolCodeExecution)],
                        ),
                    )
                response = await asyncio.to_thread(_extract_sync)
                _track_cost(response, "extraction") 
            else:
                # Text-only mode
                full_prompt = extraction_prompt + f"\n\nContent:\n{text}"

                def _extract_sync():
                    return client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(code_execution=types.ToolCodeExecution)],
                        ),
                    )
                response = await asyncio.to_thread(_extract_sync)
                _track_cost(response, "extraction")
            
            # Parse response with safety checks
            if not response.candidates:
                raise Exception("No response candidates returned by Gemini")
            
            # Check for blocking/safety issues
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason'):
                    block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                    raise Exception(f"Response blocked: {block_reason}")
            
            # Access text safely — code_execution responses have multiple parts
            response_text = None
            text_error_msgs = []

            # Gather all candidate text from every part (text + code_execution_result)
            try:
                candidate_texts: List[str] = []
                if response.candidates and len(response.candidates) > 0:
                    for part in (response.candidates[0].content.parts or []):
                        if hasattr(part, 'text') and part.text:
                            candidate_texts.append(part.text.strip())
                        elif hasattr(part, 'code_execution_result'):
                            out = getattr(part.code_execution_result, 'output', '')
                            if out:
                                candidate_texts.append(out.strip())
            except Exception as e_parts:
                text_error_msgs.append(f"parts iteration failed: {e_parts}")

            # Pick the first part that contains valid JSON (prefer arrays)
            for ct in candidate_texts:
                cleaned = ct
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()
                try:
                    json.loads(cleaned)
                    response_text = cleaned
                    break
                except (json.JSONDecodeError, ValueError):
                    continue

            # Fallback: response.text property
            if not response_text:
                try:
                    if hasattr(response, 'text') and response.text:
                        response_text = response.text.strip()
                except Exception as e1:
                    text_error_msgs.append(f"response.text failed: {str(e1)}")

            # Fallback: regex search across all candidate texts
            if not response_text:
                for ct in candidate_texts:
                    json_match = re.search(r'\[.*\]', ct, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0).strip()
                        break

            if not response_text:
                error_details = "; ".join(text_error_msgs) if text_error_msgs else "Unknown error"
                has_candidates = hasattr(response, 'candidates') and bool(response.candidates)
                raise Exception(f"Cannot access response text. Errors: {error_details}. Response type: {type(response).__name__}, has candidates: {has_candidates}")

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            try:
                extraction_result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                if "Extra data" in str(json_err):
                    bracket_count = 0
                    end_pos = 0
                    for idx_c, c in enumerate(response_text):
                        if c == '[':
                            bracket_count += 1
                        elif c == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_pos = idx_c + 1
                                break
                    if end_pos > 0:
                        extraction_result = json.loads(response_text[:end_pos])
                    else:
                        raise json_err
                else:
                    raise json_err
            
            if not isinstance(extraction_result, list):
                if isinstance(extraction_result, dict) and "speeches" in extraction_result:
                    extraction_result = extraction_result["speeches"]
                else:
                    extraction_result = []
            
            return extraction_result
            
        except Exception as e:
            error_msg = str(e)
            # Show full error message (limit to 200 chars for readability, but show more than 60)
            display_msg = error_msg if len(error_msg) <= 200 else error_msg[:200] + "..."
            print(f"  ⚠️ Attempt {attempt + 1}/{retry_attempts} failed: {display_msg}")
            if attempt < retry_attempts - 1:
                print(f"  ⏳ Retrying in 60s...")
                await asyncio.sleep(60)
            else:
                print(f"  ❌ All retry attempts failed. Full error: {error_msg}")
                return []


# Async Agent 2: Translation and Structure Analysis Agent (Async Version)
@traceable
async def async_translation_and_structure_agent(page_data: Dict[str, Any], raw_speeches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Async version: Translate speeches to English and analyze structural elements.
    """
    if not raw_speeches:
        return {"extracted_speeches": []}

    # Extract page number from metadata
    page_number = page_data.get('metadata', {}).get('page', page_data.get('page', 'unknown'))
    if page_number != 'unknown':
        try:
            page_number = int(page_number)
        except (ValueError, TypeError):
            page_number = 'unknown'

    translation_prompt = f"""You are an expert translator and analyst of Sri Lankan parliamentary Hansard debates.
Translate any Sinhala or **Sri Lankan Tamil** text to English while preserving meaning and analyzing structural elements.
Use Sri Lankan Tamil dialect, NOT Indian Tamil.
Each speech may have a "heading" field that was already extracted - preserve this.

For each speech, output an object with:
- speaker_name: The speaker's name (already in English)
- speech_text: The TRANSLATED speech text in English
- heading_text: Use the "heading" from input. If null/missing use [], if present use [heading_value]
- speech_start_time: Copy exactly from input `speech_start_time` only when the extractor provided a value; if missing, null, or empty in input, output "" — do not invent a time.
- speech_start_time_english: If `speech_start_time` is non-empty, translate it into **clear English for a clock reading** so downstream classification can tell morning vs evening. Examples: `[පූ.භා. 11.12]` → `"11:12 a.m."` or `"11:12 a.m. (forenoon)"`; use standard a.m./p.m. Interpret Sinhala/Tamil session labels (e.g. පූ.භා. / forenoon = a.m., ප.ව. / afternoon = p.m.) correctly. **One short phrase only** (time + a.m./p.m., optional brief parenthetical). If `speech_start_time` is empty, set `speech_start_time_english` to "" — do not invent a time.
- Structure flags:
  * has_tables: true if speech contains table-like content (pipe-separated data, columns, rows)

### Translation Rules:
- Translate Sinhala and **Sri Lankan Tamil** to English, keep English as-is
- **IMPORTANT**: Use Sri Lankan Tamil dialect, NOT Indian Tamil
- Do NOT translate proper nouns/names

### CRITICAL - Content Preservation During Translation:
- **DO NOT omit, skip, or remove any part of the speech during translation**
- **Go through each line carefully and translate ALL content**
- **Preserve ALL interjections, interruptions, asides, and stage directions**
- Your job is COMPLETE TRANSLATION, not summarization
- Every sentence in the original must appear in the translation

### CRITICAL - Text Formatting Rules:
- **DO NOT add newline characters (\\n) to the speech text**
- **DO NOT add forward slashes (/) or backslashes (\\) to the speech text**
- **DO NOT add any special formatting characters, line breaks, or escape sequences**
- **Keep speech text as a single continuous string with spaces between words**
- **Remove any slashes or special formatting characters that are not part of the actual content**
- The speech_text field should be plain text with only spaces separating words
- MAKE SURE YOU TRANSLATE EVERYTHING IN THE SPEECH TEXT INCLUDING EVERY REPORT OR ANYTHING SUBMITTED TO THE PARLIAMENT. DONT DROP ANYTHING. 

### Heading Handling:
- Use the "heading" field from input (already in English ALL CAPS)
- heading_text = [heading_value] if present, or [] if null/missing

### Output Format:
```json
{{
    "extracted_speeches": [
        {{
            "speaker_name": "string",
            "speech_text": "string",
            "heading_text": [],
            "speech_start_time": "",
            "speech_start_time_english": ""
        }}
    ]
}}
```

### Raw Speeches:
{json.dumps(raw_speeches, indent=2, ensure_ascii=False)}
"""

    retry_attempts = 3
    
    for attempt in range(retry_attempts):
        try:
            if client is None:
                raise Exception("Google genai client not initialized")
            
            full_prompt = translation_prompt
            
            def generate_content_sync():
                return gemini_generate_content_wrapped(
                    client_instance=client,
                    model="gemini-3-flash-preview",
                    contents=full_prompt,
                    config=None
                )

            response = await asyncio.to_thread(generate_content_sync)
            
            # Parse response with safety checks
            if not response.candidates:
                raise Exception("No response candidates returned by Gemini")
            
            # Check for blocking/safety issues
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason'):
                    block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                    raise Exception(f"Response blocked: {block_reason}")
            
            # Access text safely with multiple fallback methods
            response_text = None
            text_error_msgs = []
            
            # Method 1: Try response.text property
            try:
                if hasattr(response, 'text') and response.text:
                    response_text = response.text.strip()
            except Exception as e1:
                text_error_msgs.append(f"response.text failed: {str(e1)}")
            
            # Method 2: Try accessing text parts from candidates (skip thought parts)
            if not response_text:
                try:
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                text_parts = []
                                for part in candidate.content.parts:
                                    if hasattr(part, 'thought') and part.thought:
                                        continue
                                    if hasattr(part, 'text') and part.text:
                                        text_parts.append(part.text)
                                if text_parts:
                                    response_text = ''.join(text_parts).strip()
                except Exception as e2:
                    text_error_msgs.append(f"candidates access failed: {str(e2)}")
            
            # Method 3: Try to get any text from the response object
            if not response_text:
                try:
                    response_str = str(response)
                    import re
                    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(0).strip()
                except Exception as e3:
                    text_error_msgs.append(f"string extraction failed: {str(e3)}")
            
            if not response_text:
                error_details = "; ".join(text_error_msgs) if text_error_msgs else "Unknown error"
                has_candidates = hasattr(response, 'candidates') and bool(response.candidates)
                raise Exception(f"Cannot access response text. Errors: {error_details}. Response type: {type(response).__name__}, has candidates: {has_candidates}")
            
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            translation_result = json.loads(response_text)
            
            if not isinstance(translation_result, dict) or "extracted_speeches" not in translation_result:
                if isinstance(translation_result, list):
                    translation_result = {"extracted_speeches": translation_result}
                else:
                    translation_result = {"extracted_speeches": []}

            return translation_result
            
        except Exception as e:
            error_msg = str(e)
            display_msg = error_msg if len(error_msg) <= 200 else error_msg[:200] + "..."
            print(f"  ⚠️ Attempt {attempt + 1}/{retry_attempts} failed: {display_msg}")
            if attempt < retry_attempts - 1:
                print(f"  ⏳ Retrying in 15s...")
                await asyncio.sleep(15)
            else:
                print(f"  ❌ All retry attempts failed. Full error: {error_msg}")
                return {"extracted_speeches": []}

# Helper function to convert Dict to Pydantic models
def validate_page_data(data: Union[Dict[str, Any], PageData]) -> PageData:
    """Validate and convert page_data to PageData model"""
    if isinstance(data, PageData):
        return data
    return PageData(**data)

def validate_extracted_speech(data: Union[Dict[str, Any], ExtractedSpeech]) -> ExtractedSpeech:
    """Validate and convert extracted speech to ExtractedSpeech model"""
    if isinstance(data, ExtractedSpeech):
        return data
    if isinstance(data, dict):
        data = dict(data)
    # Handle both speech_text and speech fields
    if 'speech' in data and 'speech_text' not in data:
        data['speech_text'] = data['speech']
    if data.get('speech_start_time') is None:
        data['speech_start_time'] = ''
    if data.get('speech_start_time_english') is None:
        data['speech_start_time_english'] = ''
    return ExtractedSpeech(**data)

# Async Agent 2: Content Classification Agent (Async Version)
@traceable
async def async_content_classification_agent(page_data: Union[Dict[str, Any], PageData], extracted_speeches: Union[List[Dict[str, Any]], List[ExtractedSpeech]]) -> List[ClassifiedSpeech]:
    """
    Async version: Classifies speeches and content based on taxonomy
    Returns List[ClassifiedSpeech] with Pydantic validation
    """
    # Validate inputs
    validated_page_data = validate_page_data(page_data)
    
    if not extracted_speeches:
        return []
    
    # Convert extracted speeches to models
    validated_extracted = []
    for speech in extracted_speeches:
        validated_extracted.append(validate_extracted_speech(speech))
    
    # Extract page number from metadata
    page_number = validated_page_data.metadata.get('page', validated_page_data.page)
    if page_number != 'unknown':
        try:
            page_number = int(page_number)
        except (ValueError, TypeError):
            page_number = 'unknown'

    # Prepare speeches with their existing structure flags for classification
    # NOTE: We DO NOT send original_speech to the LLM - it might change it
    speeches_for_classification = []
    for speech in validated_extracted:
        speech_text = speech.speech_text or speech.speech or ""
        structure_flags = speech.structure_flags or StructureFlags()
        speech_with_flags = {
            "speaker_name": speech.speaker_name or "",
            "speech_text": speech_text,
            "structure_flags": {
                "has_tables": structure_flags.has_tables,
                "heading_text": structure_flags.heading_text if structure_flags.heading_text else speech.heading_text
            }
            # original_speech is NOT sent to classification agent
        }
        speeches_for_classification.append(speech_with_flags)

    classification_prompt = f"""You are an expert analyst of Sri Lankan parliamentary Hansard debates.
Your task is to classify **already extracted speeches** into predefined thematic categories.
Always return output strictly as a **valid JSON array** (no extra text outside JSON).

---

### Processing Rules

For each extracted speech, output an object with the following fields:

- **page**: {page_number} (use this exact page number)
- **speaker_name**: use the speaker_name from the extracted data
- **speech**: use the speech_text from the extracted data
- **theme**:
  - main_category: must be one of the 15 top-level categories in {TAXONOMY}
  - subcategory: the most relevant subcategory from {TAXONOMY}
  - justification: 1–2 sentences explaining the reasoning for this classification
- **structure_flags**: use the existing structure_flags from the extracted speech data (do not regenerate)

---

### Taxonomy

{TAXONOMY}

---

### Output Format

Always return in this format (array of objects):

```json
[
  {{
    "page": {page_number},
    "speaker_name": "string",
    "speech": "string",
    "theme": {{
      "main_category": "string",
      "subcategory": "string",
      "justification": "string"
    }},
    "structure_flags": {{
      "has_tables": false,
      "heading_text": ["string"]
    }}
  }}
]
```

### Important Notes
- Do not invent or alter the extracted speech text. Use exactly what is given.
- Use the existing structure_flags from the extracted speech data - do not regenerate them.
- Classify using the taxonomy in {TAXONOMY}. Both main_category and subcategory must match exactly.
- Keep justification concise (1–2 sentences).
- Output only JSON (no explanations, no markdown, no commentary).

### Extracted Speeches to Classify:
{json.dumps(speeches_for_classification, indent=2)}
"""

    # Retry logic: 2 retry attempts after 60 seconds
    retry_attempts = 3  # Initial attempt + 2 retries
    
    for attempt in range(retry_attempts):
        try:
            if client is None:
                raise Exception("Google genai client not initialized")
            
            # Combine system and user messages
            full_prompt = f"You are an expert at classifying parliamentary content. Return only valid JSON.\n\n{classification_prompt}"
            
            # Use Google genai client with LangSmith wrapper (run in thread pool for async compatibility)
            def generate_content_sync():
                return gemini_generate_content_wrapped(
                    client_instance=client,
                    model="gemini-3-flash-preview",
                    contents=full_prompt,
                    config=None
                )
            
            response = await asyncio.to_thread(generate_content_sync)
            
            # Safely extract response text (access text parts directly to avoid thought_signature warnings)
            response_text = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        # Extract only text parts, ignore thought_signature
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        if text_parts:
                            response_text = ''.join(text_parts).strip()
            
            # Fallback to response.text if direct access didn't work
            if not response_text:
                if hasattr(response, 'text') and response.text:
                    response_text = response.text.strip()
                else:
                    raise Exception("Empty response from Gemini")
            
            # Handle markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Validate JSON before parsing
            if not response_text or not response_text.strip():
                raise Exception("Empty or whitespace-only response text after cleaning")
            
            # Try to parse JSON with better error handling
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                error_msg = str(json_err)
                # Handle "Expecting value" (empty JSON) error
                if "Expecting value" in error_msg or "line 1 column 1" in error_msg:
                    print(f"  [WARNING] Empty or invalid JSON response, returning empty list")
                    result = []
                # Try to extract just the first valid JSON object/array if there's extra data
                elif "Extra data" in error_msg:
                    # Find the position where extra data starts
                    try:
                        # Try to find the first complete JSON array/object
                        if response_text.strip().startswith('['):
                            # Find the matching closing bracket
                            bracket_count = 0
                            end_pos = 0
                            for i, char in enumerate(response_text):
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end_pos = i + 1
                                        break
                            if end_pos > 0:
                                result = json.loads(response_text[:end_pos])
                            else:
                                raise json_err
                        elif response_text.strip().startswith('{'):
                            # Find the matching closing brace
                            brace_count = 0
                            end_pos = 0
                            for i, char in enumerate(response_text):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_pos = i + 1
                                        break
                            if end_pos > 0:
                                result = json.loads(response_text[:end_pos])
                            else:
                                raise json_err
                        else:
                            raise json_err
                    except:
                        raise json_err
                else:
                    raise json_err

            # Ensure result is a list
            if isinstance(result, list):
                classified_speeches = result
            elif "speeches" in result:
                classified_speeches = result["speeches"]
            else:
                classified_speeches = []

            # Validate each speech and merge with original structure flags
            validated_speeches = []
            for i, speech in enumerate(classified_speeches):
                try:
                    # Get the original structure flags from the extracted speech
                    original_speech = validated_extracted[i] if i < len(validated_extracted) else None
                    
                    # Normalize heading_text to ensure it's always a list
                    if original_speech:
                        heading_text = normalize_heading_text(
                            original_speech.structure_flags.heading_text if original_speech.structure_flags 
                            else original_speech.heading_text
                        )
                        original_structure_flags = StructureFlags(
                            has_tables=original_speech.structure_flags.has_tables if original_speech.structure_flags else False,
                            heading_text=heading_text
                        )
                        original_speech_text = original_speech.original_speech
                    else:
                        heading_text = normalize_heading_text(speech.get('structure_flags', {}).get('heading_text', []))
                        original_structure_flags = StructureFlags(
                            has_tables=speech.get('structure_flags', {}).get('has_tables', False),
                            heading_text=heading_text
                        )
                        original_speech_text = ""

                    # Use the original structure flags, not the LLM-generated ones
                    speech['structure_flags'] = original_structure_flags.model_dump()
                    
                    # Preserve original_speech from the extracted speech
                    speech['original_speech'] = original_speech_text
                    speech['speech_start_time'] = (
                        (original_speech.speech_start_time or "") if original_speech else ""
                    )
                    speech['speech_start_time_english'] = (
                        (original_speech.speech_start_time_english or "") if original_speech else ""
                    )

                    # Set default input_id and input_type (will be assigned by Agent 3)
                    speech['input_id'] = 0
                    speech['input_type'] = ""
                    speech['has_petition'] = "no"
                    speech['petition_count'] = 0
                    speech['has_expunged'] = "no"
                    speech['expunged_count'] = 0
                    
                    # Ensure speaker_name is not None (convert to empty string for Pydantic)
                    if speech.get('speaker_name') is None:
                        speech['speaker_name'] = ""

                    validated_speech = ClassifiedSpeech(**speech)
                    validated_speeches.append(validated_speech)
                except Exception as validation_error:
                    print(f"  [WARNING] Speech validation failed: {validation_error}")
                    # Try to create a minimal valid speech if validation fails
                    try:
                        minimal_speech = ClassifiedSpeech(
                            page=page_number if isinstance(page_number, int) else 0,
                            speaker_name="",
                            speech=speech.get('speech', ''),
                            theme=Theme(
                                main_category=speech.get('theme', {}).get('main_category', ''),
                                subcategory=speech.get('theme', {}).get('subcategory', ''),
                                justification=speech.get('theme', {}).get('justification', '')
                            ),
                            structure_flags=StructureFlags(),
                            input_id=0,
                            input_type="",
                            original_speech="",
                            speech_start_time="",
                            speech_start_time_english="",
                        )
                        validated_speeches.append(minimal_speech)
                    except:
                        print(f"  [ERROR] Could not create minimal speech, skipping")

            return validated_speeches
            
        except Exception as e:
            error_msg = str(e)
            # Show full error message (limit to 200 chars for readability)
            display_msg = error_msg if len(error_msg) <= 200 else error_msg[:200] + "..."
            print(f"  ⚠️ Attempt {attempt + 1}/{retry_attempts} failed: {display_msg}")
            if attempt < retry_attempts - 1:
                print(f"  ⏳ Retrying in 60s...")
                await asyncio.sleep(60)
            else:
                print(f"  ❌ All retry attempts failed. Full error: {error_msg}")
                return []

# Helper function to generate input ID classification prompt
def generate_input_id_prompt(page_number: int, classified_speeches: List[Dict[str, Any]], context_speeches: List[Dict[str, Any]] = None) -> str:
    """
    Generate the input ID classification prompt for both sync and async agents.
    Optional context_speeches: last N speeches from the previous page, sent as read-only
    context so the LLM can classify cross-page patterns correctly.
    """
    context_block = ""
    if context_speeches:
        # Strip original_speech to keep token count down
        slim_context = []
        for s in context_speeches:
            slim_context.append({
                "page": s.get("page", ""),
                "speaker_name": s.get("speaker_name", ""),
                "speech_start_time": s.get("speech_start_time", ""),
                "speech_start_time_english": s.get("speech_start_time_english", ""),
                "speech": (s.get("speech", "")[:200] + "...") if len(s.get("speech", "")) > 200 else s.get("speech", ""),
                "input_id": s.get("input_id", 0),
                "input_type": s.get("input_type", ""),
                "structure_flags": s.get("structure_flags", {})
            })
        context_block = f"""
### Context from Previous Page (READ-ONLY — DO NOT include in output)
The following speeches are from the PREVIOUS page, already classified.
Use them to understand the ongoing parliamentary context when classifying the current page.
Do NOT output these speeches — they are for reference only.

Use this context to handle patterns that span page boundaries, such as:
- If the last context speech is a question (ID 1, 3, 17, 19, 24), the first matching
  response speech on this page should receive the corresponding answer ID (2, 4, 18, 23, 25).
- If the last context speech declares intent (e.g. "I rise to a point of order",
  "I move that...", "I wish to raise a matter of privilege"), the first relevant
  speech on this page by the same or responding speaker continues that action
  and should be classified accordingly (ID 8/9 for point of order, ID 16 for privilege, etc.).
- If a bill debate or adjournment discussion is ongoing, continue classifying under the same ID (e.g. live Motion at Adjournment Time → **ID 10** can continue past 5:00 p.m. until the topic/section clearly shifts to forthcoming business → **ID 28**).
- If a follow-up question sequence is in progress, apply the correct follow-up IDs.

{json.dumps(slim_context, indent=2)}

---
"""

    return f"""You are an expert classifier of Sri Lankan Parliamentary Hansard proceedings.
Your task is to classify each Hansard entry (speech, question, or note) by matching it to the most appropriate input_id and input_type.
Use the English all-caps heading of the Hansard section to guide classification when relevant.READ THE SPEECH FROM THE START TO THE END.
Return the output in the exact JSON structure shown at the end.
Always return output strictly as a **valid JSON array** (no extra text outside JSON).

---

### Hansard speech time (ID 10 vs ID 28 only)
Each speech may include **`speech_start_time`** (original bracketed line) and **`speech_start_time_english`** (clock in English, e.g. `"5:42 p.m."`). **Classification is still driven by the heading and speech content first** — do not let time override a clear section heading for any ID.

**Use `speech_start_time_english` for one purpose only:** when the heading (and context) could reasonably match **either** the live **Motion at Adjournment Time** debate (**ID 10**) **or** the **forthcoming business / advance agenda** adjournment motion (**ID 28**), use the time as a **tie-breaker** — typically **after 5:00 p.m.** supports **ID 10**; **at or before 5:00 p.m.** supports **ID 28**. **However:** a live adjournment-time debate **can continue past 5:00 p.m.** — if **context** shows that debate is **still ongoing** (same topic/section), keep **ID 10**; use **ID 28** when the **topic/section** has **clearly shifted** to forthcoming business (see SECTION 4). If `speech_start_time_english` is empty, choose between 10 and 28 from heading wording, context, and content only; **do not invent a time**. **Do not** use this field to pick or exclude any other input_id (e.g. ADJOURNMENT QUESTIONS → still **24/25** from the heading alone).

---

### SECTION 1 – ORAL ANSWERS TO QUESTIONS
(Apply only if the heading is "ORAL ANSWERS TO QUESTIONS")

**ID 1: Written Question**
- MP asks a question verbally or in writing
- Rule A: Use only under "ORAL ANSWERS TO QUESTIONS"

**ID 2: Written Question – Response**
- Minister responds orally to a question
- Rule A: Use ONLY under "ORAL ANSWERS TO QUESTIONS" — **NEVER use ID 2 under "QUESTION BY PRIVATE NOTICE"**
- If the heading is "QUESTION BY PRIVATE NOTICE", a minister's response must use ID 18, not ID 2
- In the speech, if they say they'll give the answer later, then it's classified as oral contribution with ID 14.
- CHECK THE RELEVANCY OF THE ANSWER. If the relevant answer is not given to the question, record under id 14. 

**ID 3: Written Question – Follow-Up Question**
- MP asks a follow-up after the initial reply
- Rule A: Use only under "ORAL ANSWERS TO QUESTIONS"

**ID 4: Follow-Up Question – Response**
- Minister answers a follow-up question
- Rule A: Use only under "ORAL ANSWERS TO QUESTIONS"
- In the speech, if they say they'll give the answer later, then it's classified as oral contribution with ID 14.
- CHECK THE RELEVANCY OF THE ANSWER. If the relevant answer is not given to the question, record under id 14. 


**ID 5: Written Responses**
- Minister provides a written answer printed at the end of the Hansard
- The original question is under ID 1
- Rule A: Use only under "ORAL ANSWERS TO QUESTIONS"
 

---

### SECTION 2 – BILLS PRESENTED

**ID 6: Bill / Regulation / Order / Resolution – Administration**
- Administrative or procedural introduction of a Bill, Regulation, Order, or Resolution (e.g. "I beg to move...", "The Bill is presented", clerk reading the title, formal first reading)
- Rule B: Use only under "BILLS PRESENTED" — and ONLY for the administrative/procedural intro speech itself, NOT for debate speeches. Dont use this if the heading is just mentioning BILLS in one part. If so it should be classified as ID 7. Use this only under BILLS PRESENTED.

---

### SECTION 3 – GENERAL (CROSS-CUTTING ITEMS)

**ID 7: Bill / Regulation / Order / Resolution – Debate Oral Contribution**
- **ANY speech made during a debate on a Bill, Regulation, Order, Resolution, or Private Incorporation Bill**
- This includes: second-reading speeches, third-reading speeches, committee-stage speeches, amendments, any MP speaking for/against/about a Bill
- **CRITICAL**: When the heading contains a Bill name (e.g. "NATIONAL BUILDING RESEARCH INSTITUTE BILL", "FINANCE BILL", "URBAN DEVELOPMENT AUTHORITY (AMENDMENT) BILL") OR when heading is "BILLS PRESENTED" and the speech is actual debate (not administrative introduction), use ID 7
- **DO NOT use ID 14 for speeches under a bill heading** — use ID 7 instead
- Rule C: Can appear under any heading; detect via content cues such as debate or bill heading

**ID 8: Point of Order – Technical / Procedural**
- Valid procedural issue raised under Standing Order 71(3)
- **CRITICAL**: When a speaker says "I rise to a point of order" or "මට රීති ප්‍රශ්නයක් මතු කරන්න ඕන", this is just the REQUEST to raise a point of order
- The ACTUAL point of order is the NEXT speech by the SAME speaker where they explain the procedural issue
- DO NOT classify the speech containing "I rise to a point of order" as ID 8 - classify the NEXT speech by that speaker instead
- Rule C: Can appear under any heading
- Pay attention to the context of above speeches when classifying as a point of order

**ID 9: Point of Order – Other**
- Non-procedural or irrelevant point acknowledged as not a point of order by the Speaker
- **CRITICAL**: Same logic as ID 8 - the actual point is in the NEXT speech, not the "I rise to a point of order" statement.After a point of oder is raised, it would be rejected by any person in the next speech.
- ** Use this only when the point of order is rejected by any person.Else it should be classified as ID 8.
- Rule C: Can appear under any heading
- pay attention to the context of speeches beefore when classifying this. 

---

### SECTION 4 – ADJOURNMENT MOTIONS

**ID 10: Motion at Adjournment Time – Oral Contribution**
- **ANY speech made during an adjournment-time debate on a matter of urgent public importance**
- This includes: all MP speeches, minister responses, interjections — anything spoken during the adjournment debate
- **CRITICAL (heading first)**: When the heading is "ADJOURNMENT MOTION", "MOTION AT ADJOURNMENT TIME", or similar **live adjournment-time debate** wording, ALL speeches in that debate → **ID 10** — DO NOT use ID 14. **The heading and section context remain the main decision** — keep applying Rule D from the heading as usual.
- **Debate can run past 5:00 p.m.**: A live Motion at Adjournment Time debate **often continues after 5:00 p.m.** Do **not** treat the clock alone as proof the speech is **ID 28**.
- **Use context (previous page)**: If **Context from Previous Page** shows an **ongoing** evening adjournment-time debate (same matter, same debate flow) and this page **continues** that debate, classify → **ID 10** — even if `speech_start_time_english` is after 5:00 p.m.
- **Switch to ID 28 after ~5:00 p.m. when the topic/section changes**: If **around or after approximately 5:00 p.m.** the Hansard **clearly moves on** — new **forthcoming business / advance agenda** adjournment item, **different** adjournment subject, or context shows the **live debate has ended** and the record is now **forthcoming business** — use **ID 28** for that material, not ID 10.
- **ID 10 vs ID 28 (tie-break only)**: When heading **and** context still do **not** distinguish live debate vs forthcoming business, use `speech_start_time_english`: **after 5:00 p.m.** → lean **ID 10**; **at or before 5:00 p.m.** → lean **ID 28**. If the time field is empty, use heading phrasing (debate vs forthcoming business) and speech content — do not guess a time.
- Usually near the end of a sitting
- Rule D: Topic = ADJOURNMENT; phrases like "Motion at Adjournment Time" or "Adjournment Debate"

---

### SECTION 5 – PETITIONS

**ID 11: Petitions**
- Formal petitions submitted by MPs (2–3 lines)
- Count and record number of lines separately
- Rule E: Use only under heading "PETITIONS"
- **IMPORTANT**: If this speech is classified as ID 11, set "has_petition" to "yes" in the output
- **IMPORTANT**: Count the number of individual petitions mentioned/submitted within the speech and set "petition_count" to that number. A single speech may contain multiple petitions (e.g. "I hereby submit 3 petitions..." or multiple petition references). If the speech mentions a specific number, use that. If it lists petitions individually, count them. Default to 1 if it is a petition but no specific count is discernible.

---

### SECTION 6 – PROCEDURAL / BEHAVIOURAL

**ID 13: Expunged Statement**
- Statement ordered removed from the record
- **CRITICAL**: Use ID 13 ONLY if the ENTIRE speech is expunged (the whole speech content is removed/expunged)
- If only a small part or portion of the speech contains expunged content, DO NOT use ID 13
- Instead, classify it with the appropriate input_id based on the actual content (e.g., ID 14 for Oral Contribution, ID 1 for Written Question, etc.)
- Record one line per expunged item
- Look for: "[Expunged]", "Expunged on the order of..."
- Rule F: Can apply anywhere

**ID 14: Oral Contribution**
- General speech or remark not covered by any other ID
- Rule F: Can apply anywhere; use only when no other specific ID fits; this is the DEFAULT fallback
- Additionally, if the heading has REPORT in it, then it should be classified as ID 14.
- There's another case where the heading AUDITOR GENERAL'S REPORT is present, then the report submitted by the speaker should be classified as ID 14.
- Cases where a request is being made to give an answer to a question later should be recorded under this.

---

### SECTION 7 – CORE STATEMENTS & ANNOUNCEMENTS

**ID 15: Oral Contribution – Core Statements**
- Significant oral statements or announcements
- Examples: ministerial statements, personal clarifications, votes of condolence
- Rule G: Applies across headings; look for "STATEMENT BY MINISTER…" or "VOTES OF CONDOLENCE"

---

### SECTION 8 – PARLIAMENTARY PRIVILEGES

**ID 16: Parliamentary Privileges**
- When an MP raises an issue about breach or violation of parliamentary privilege
- Example: "PRIVILEGE: Legal action against evidence given at COPA"
- Rule H: No topic restriction; heading often starts with "PRIVILEGE"

---

### SECTION 9 – QUESTION BY PRIVATE NOTICE

**ID 17: Question by Private Notice**
- Urgent question raised by a party leader or MP requiring immediate answer
- Rule I: Use only under "QUESTION BY PRIVATE NOTICE"
- Only the First question after the main heading should be classified as ID 17.
- Other follow up questions should be classified as oral contribution with ID 14.

**ID 18: Question by Private Notice – Answer**
- Minister's response to a private-notice question
- **CRITICAL**: This is the ONLY correct ID for any ministerial or government response under "QUESTION BY PRIVATE NOTICE"
- **NEVER use ID 2 (Written Question – Response) for responses under this heading** — ID 2 is exclusively for "ORAL ANSWERS TO QUESTIONS"
- Rule I: Use only under "QUESTION BY PRIVATE NOTICE"
- Only the First answer after the main heading should be classified as ID 17.
- Other answers tofollow up questions should be classified as oral contribution with ID 14.
- In the speech, if they start mentioning they are answering a question  under Standing Order 27 (2), then it should be classified as ID 18.
---

### SECTION 10 – QUESTIONS POSED TO HON. PRIME MINISTER

**ID 19: Prime Minister's Questions (PMQ)**
- Main questions asked to the PM during PM Question Time
- Rule J: Use only under "QUESTIONS POSED TO HON. PRIME MINISTER"

**ID 19-1 / 19-2: PM Questions – Follow-Up Questions**
- Follow-up questions asked after the main PM question
- **19-1** = the **first** follow-up question asked (after the main PM question)
- **19-2** = the **second** follow-up question asked
- **Constraint**: Only **two** follow-up questions are allowed per MP under PMQ (never go beyond 19-2)
- Rule J: Use only under "QUESTIONS POSED TO HON. PRIME MINISTER"

**ID 23: PMQ-Responses**
- PM or Minister's main answers to PM questions
- Rule J: Use only under "QUESTIONS POSED TO HON. PRIME MINISTER"

**ID 23-1 / 23-2: PM Questions – Follow-Up Answers**
- PM or Minister's answers to follow-up questions
- **23-1** = answer to the **first** follow-up question (answers 19-1)
- **23-2** = answer to the **second** follow-up question (answers 19-2)
- Rule J: Use only under "QUESTIONS POSED TO HON. PRIME MINISTER"

---

### SECTION 11 – PAPERS PRESENTED

**ID 20: Notification and Paper Presentations**
- Notifications or papers formally laid on the Table
- Examples: reports, gazette notices, committee papers
- Rule K: Use only under "PAPERS PRESENTED"

---

### SECTION 12 – ANNOUNCEMENTS

**ID 21: Announcement**
- Announcements made by the Speaker or Deputy Speaker
- Usually at the start of a sitting
- Rule L: Use only under "ANNOUNCEMENTS"

---

### SECTION 13 – PRIVATE MEMBERS' MOTIONS

**ID 22: Private Member Motions (PMM)**
- Motions and related debate introduced by non-minister MPs
- Rule M: Use only under "PRIVATE MEMBERS' MOTIONS"

---

### SECTION 14 – ADJOURNMENT QUESTIONS

**ID 24: Adjournment Question**
- Questions asked at adjournment time on urgent matters
- Rule N: Use only under "ADJOURNMENT QUESTIONS"

**ID 25: Adjournment Question – Responses**
- Minister's responses to adjournment questions
- Rule N: Use only under "ADJOURNMENT QUESTIONS"

---

### SECTION 15 – PRIVATE MEMBERS' BILLS

**ID 26: Private Member's Bill – Public Interest**
- Private Member's Bill intended for public benefit
- Note: Incorporation-type private bills remain under ID 6
- Rule O: Use only under "PRIVATE MEMBERS' BILLS"

---

### SECTION 16 – BUDGET SPEECH

**ID 27: Budget Speech**
- The national budget speech delivered by the Minister of Finance
- Outlines fiscal policy and expenditure plans
- Rule P: Use only under "BUDGET SPEECH"

---

### SECTION 17 – ADJOURNMENT MOTION (FORTHCOMING BUSINESS)

**ID 28: Adjournment Motion**
- Motion to adjourn proceedings on an issue of urgent public importance
- Identified from the "Forthcoming Business" of the House
- **CRITICAL (heading first)**: When the heading/context clearly indicates **forthcoming business** / advance agenda (not the live evening adjournment debate), use **ID 28** — DO NOT use ID 14. **The heading remains the main signal** for Section 17.
- **ID 10 vs ID 28 only**: If the heading could fit **either** live debate or forthcoming business, prefer **context** (ongoing live debate → **ID 10**; clear shift to forthcoming business after ~5:00 p.m. → **ID 28**). If still ambiguous, use `speech_start_time_english` as a **tie-breaker** — **not after 5:00 p.m.** → lean **ID 28**; **clearly after 5:00 p.m.** → lean **ID 10**. If the time field is empty, disambiguate from heading wording and content only.
- Rule Q: Topic = ADJOURNMENT; same subject matter as ID 10 but sourced from advance agenda / forthcoming business

---

### SECTION 18 – PRESIDENT'S SPEECH

**ID 29: President's Speech**
- Speeches delivered by the President
- Special ceremonial addresses

---

### SECTION 19 – ADDRESS

**ID 30: Address**
- When a speaker (typically the Speaker, Deputy Speaker, or Chairperson) calls out someone by name in parliament, asking or inviting them to speak
- Examples: "Hon. Wijesiri Basnayake", "The Hon. Minister of Finance", "I call upon the Hon. Member..."
- When speaker annonces someone coming into the parliament.
- These are typically very short — just the name or a brief invitation to speak
- Rule S: Can appear under any heading; detect via content (speaker calling out or naming another member to speak)

---

### Classification Strategy

For each speech:
1. **Check section heading** first (if available in structure_flags.heading_text):
   - Match the heading to the appropriate SECTION above
   - Apply the corresponding Rule (A through Q)

   **HEADING-BASED MANDATORY OVERRIDES** (these take priority over everything else):
   - If heading contains a **Bill name** (e.g. ends with "BILL", "ACT", "REGULATION", "ORDER", "RESOLUTION") OR heading is "BILLS PRESENTED":
     * Administrative intro speech (first reading, "I beg to move", clerk reading title) → ID 6
     * ALL other speeches in the same section (debate, for/against, amendments, responses) → **ID 7** (NOT ID 14)
   - **Adjournment: ID 10 vs ID 28 (heading first; context before raw time)**:
     * Decide from the **heading**, **Context from Previous Page**, and **whether the same live debate continues** vs **forthcoming business**: **live ongoing debate** → **ID 10** (even past 5:00 p.m.); **clear topic/section shift to forthcoming business** (often around/after ~5:00 p.m.) → **ID 28**. **Do not override a clear heading with the clock.**
     * **Only if** heading **and** context are **still ambiguous** between ID 10 and ID 28, use `speech_start_time_english`: **after 5:00 p.m.** → prefer **ID 10**; **at or before 5:00 p.m.** → prefer **ID 28**.
     * If `speech_start_time_english` is **empty**, use heading, context, and content only; do not invent a time.
     * Under the correct adjournment heading, ALL such speeches → **ID 10** or **ID 28** as above (NOT ID 14).
   - If heading is "QUESTION BY PRIVATE NOTICE":
     * MP asking the question → **ID 17**
     * ANY minister/government response → **ID 18** (NOT ID 2)
     * Follow-up questions by MPs → **ID 17**
     * Follow-up responses by ministers → **ID 18**
     * **STRICT RULE**: ID 2 ("Written Question – Response") is BANNED under this heading
   - **RULE**: Under a bill, adjournment, or QPN heading, always apply the section-specific IDs — never fall back to ID 14 or ID 2

2. **SPECIAL HANDLING FOR "I RISE TO A POINT OF ORDER"**:
   - If a speech contains "I rise to a point of order" or similar phrases, this is just the REQUEST
   - DO NOT classify this speech as ID 8 or ID 9
   - Instead, look at the NEXT speech by the SAME speaker - that is where the actual point of order is explained
   - Classify the NEXT speech (not the current one) as ID 8 or ID 9 based on the content
   - ID 9 only comes up when the point of order is rejected by any person after the point of order is raised.
   - The speech with "I rise to a point of order" should be classified as ID 14 (Oral Contribution) or another appropriate ID
   - **NOTE**: For point of order flags, apply to the NEXT speech after "I rise to a point of order", not the request itself

4. **Check speaker_name and speech content**:
   - Questions (interrogative format) → IDs 1, 3, 17, 19, 24 (use 17 only under QPN heading, use 1/3 only under ORAL ANSWERS heading)
   - Minister responses → use heading to decide: under "ORAL ANSWERS TO QUESTIONS" → IDs 2, 4, 5; under "QUESTION BY PRIVATE NOTICE" → ID 18; under PMQ → IDs 23, 25
   - Bill/debate language → IDs 6, 7
   - Motion language → IDs 10, 22, 28
   - Announcements by Speaker → ID 21
   - Budget language → ID 27
   - President speaking → ID 29

5. **Default fallback**:
   - If no specific category fits → ID 14 (Oral Contribution)
   - If it's a significant/important statement → ID 15 (Oral Contribution – Core Statements)
   - **EXCEPTION**: ID 14 must NOT be used when the heading is a bill name or adjournment motion — use ID 7 or ID 10/28 respectively

---

### Input ID to Input Type Mapping

Use these exact mappings:
1: "Written Question"
2: "Written Question - Response"
3: "Written Question - Follow up question"
4: "Follow up question - Response"
5: "Written Responses"
6: "Bill / Regulation / Order /Resolution - Administration"
7: "Bill / Regulation / Order /Resolution - Debate Oral Contribution"
8: "Point of Order- Technical/Procedural"
9: "Point of Order - Other"
10: "Motion at the Adjournment Time"
11: "Petitions"
13: "Expunged Statement"
14: "Oral Contribution"
15: "Oral Contribution - Core Statements"
16: "privileges"
17: "Question by private notice"
18: "Question by private notice - Ans"
19: "PMQ"
"19-1": "PMQ - Follow-Up Question (1st)"
"19-2": "PMQ - Follow-Up Question (2nd)"
20: "Notification"
21: "Announcement"
22: "PMM"
23: "PMQ-Responses"
"23-1": "PMQ-Responses - Follow-Up Answer (1st)"
"23-2": "PMQ-Responses - Follow-Up Answer (2nd)"
24: "Adjournment Question"
25: "Adjournment Question-Responses"
26: "Private Member's Bill - Public interest"
27: "Budget Speech"
28: "Adjournment Motion"
29: "President's Speech"

---
**IMPORTANT**: For each speech:
- If input_id is 11 (Petitions), set "has_petition" to "yes", otherwise set it to "no"
- If input_id is 11 (Petitions), set "petition_count" to the number of individual petitions submitted within the speech (minimum 1). Otherwise set it to 0.
- For "has_expunged": 
  * Set "has_expunged" to "yes" if the speech contains ANY expunged content (even if only a small part)
  * Set "has_expunged" to "no" if there is no expunged content in the speech
  * **NOTE**: This flag indicates the presence of expunged content, regardless of the input_id
  * If the ENTIRE speech is expunged, use input_id 13 AND set has_expunged to "yes"
  * If only PART of the speech is expunged, use the appropriate input_id (NOT 13) BUT still set has_expunged to "yes"
- Set "expunged_count" to how many distinct expunged statements or expunged segments appear in that speech (0 if has_expunged is "no", otherwise at least 1).

### Output Format

Return the speeches with assigned input_id and input_type:

```json
[
  {{
    "page": {page_number},
    "speaker_name": "string",
    "speech_start_time": "",
    "speech_start_time_english": "",
    "speech": "string",
    "theme": {{
      "main_category": "string",
      "subcategory": "string",
      "justification": "string"
    }},
    "structure_flags": {{
      "has_tables": false,
      "heading_text": ["string"]
    }},
    "input_id": 1,
    "input_type": "Written Question",
    "has_petition": "no",
    "petition_count": 0,
    "has_expunged": "no",
    "expunged_count": 0
  }}
]
```


---

### Important Notes
- Use structure_flags as strong hints for classification
- Preserve all existing data (page, speaker_name, speech_start_time, speech_start_time_english, speech, theme)
- Only add/update input_id and input_type fields
- Choose the most specific and appropriate input_id based on the rules above
- When in doubt between two IDs, choose the more specific one
- Output only JSON (no explanations, no markdown, no commentary)
{context_block}
### Speeches to Classify:
Each object below includes **`speech_start_time`** and **`speech_start_time_english` immediately after `speaker_name`** (before the long `speech` text). You receive these fields — **read them** when applying ID 10 vs ID 28 tie-break rules.

{json.dumps(classified_speeches, indent=2, ensure_ascii=False)}
"""

# Helper to convert ClassifiedSpeech to dict for JSON serialization
def classified_speech_to_dict(speech: Union[ClassifiedSpeech, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert ClassifiedSpeech to dict for JSON serialization"""
    if isinstance(speech, ClassifiedSpeech):
        return speech.model_dump()
    return speech


def classified_speech_to_dict_for_input_id_prompt(speech: Union[ClassifiedSpeech, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Same data as classified_speech_to_dict, but key order optimized for the input-ID LLM:
    Pydantic model_dump() puts huge `speech`/`theme` before `speech_start_time_*`, so the model
    often misses times at the end of the JSON. Put clock fields right after speaker_name.
    """
    d = classified_speech_to_dict(speech)
    priority = (
        "page",
        "speaker_name",
        "speech_start_time",
        "speech_start_time_english",
        "structure_flags",
        "theme",
        "speech",
        "original_speech",
        "input_id",
        "input_type",
        "has_petition",
        "petition_count",
        "has_expunged",
        "expunged_count",
        "was_answered",
        "word_count",
        "pages",
        "is_aggregated",
        "aggregated_count",
    )
    out: Dict[str, Any] = {}
    for k in priority:
        if k in d:
            out[k] = d[k]
    for k, v in d.items():
        if k not in out:
            out[k] = v
    return out

# Async Agent 3: Input ID Classification Agent (Async Version)
@traceable
async def async_input_id_classification_agent(page_data: Union[Dict[str, Any], PageData], classified_speeches: Union[List[Dict[str, Any]], List[ClassifiedSpeech]], context_speeches: Union[List[Dict[str, Any]], List[ClassifiedSpeech], None] = None) -> List[ClassifiedSpeech]:
    """
    Async version: Assigns input_id and input_type to classified speeches.
    Optional context_speeches: last N speeches from the previous page passed as
    read-only context for cross-page classification patterns.
    Returns List[ClassifiedSpeech] with Pydantic validation.
    """
    if not classified_speeches:
        return []

    # Convert to ClassifiedSpeech models if needed
    validated_speeches = []
    for speech in classified_speeches:
        if isinstance(speech, ClassifiedSpeech):
            validated_speeches.append(speech)
        else:
            validated_speeches.append(ClassifiedSpeech(**speech))
    
    # Extract page number from the first speech
    page_number = validated_speeches[0].page if validated_speeches else 'unknown'
    
    # Reorder keys so time fields are not buried after huge speech JSON (model was missing them)
    classified_speeches_dict = [classified_speech_to_dict_for_input_id_prompt(s) for s in validated_speeches]
    context_speeches_dict = (
        [classified_speech_to_dict_for_input_id_prompt(s) for s in context_speeches]
        if context_speeches
        else None
    )

    # Generate prompt using helper function (same as sync version)
    input_id_prompt = generate_input_id_prompt(page_number, classified_speeches_dict, context_speeches_dict)

    # Retry logic: 2 retry attempts after 60 seconds
    retry_attempts = 3  # Initial attempt + 2 retries
    
    for attempt in range(retry_attempts):
        try:
            if client is None:
                raise Exception("Google genai client not initialized")
            
            # Combine system and user messages
            full_prompt = f"You are an expert at classifying parliamentary speech types. Return only valid JSON.\n\n{input_id_prompt}"
            
            # Use Google genai client with LangSmith wrapper (run in thread pool for async compatibility)
            def generate_content_sync():
                return gemini_generate_content_wrapped(
                    client_instance=client,
                    model="gemini-2.5-pro",
                    contents=full_prompt,
                    config=None
                )
            
            response = await asyncio.to_thread(generate_content_sync)

            # Get response and clean it for JSON parsing (access text parts directly to avoid thought_signature warnings)
            response_text = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        # Extract only text parts, ignore thought_signature
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        if text_parts:
                            response_text = ''.join(text_parts).strip()
            
            # Fallback to response.text if direct access didn't work
            if not response_text:
                if hasattr(response, 'text') and response.text:
                    response_text = response.text.strip()
                else:
                    raise Exception("Empty response from Gemini")
            
            # Remove markdown code blocks if present
            if response_text.strip().startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Validate response before parsing
            if not response_text or not response_text.strip():
                print(f"  [WARNING] Empty response text, returning original speeches")
                return validated_speeches
            
            # Try to parse JSON with error handling for invalid escape sequences and extra data
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                error_msg = str(json_err)
                # Handle "Expecting value" (empty JSON) error
                if "Expecting value" in error_msg or "line 1 column 1" in error_msg:
                    print(f"  [WARNING] Empty or invalid JSON response, returning original speeches")
                    return validated_speeches
                # Handle "Extra data" errors by extracting first valid JSON
                elif "Extra data" in error_msg:
                    try:
                        # Try to find the first complete JSON array/object
                        if response_text.strip().startswith('['):
                            # Find the matching closing bracket
                            bracket_count = 0
                            end_pos = 0
                            for i, char in enumerate(response_text):
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end_pos = i + 1
                                        break
                            if end_pos > 0:
                                result = json.loads(response_text[:end_pos])
                            else:
                                raise json_err
                        elif response_text.strip().startswith('{'):
                            # Find the matching closing brace
                            brace_count = 0
                            end_pos = 0
                            for i, char in enumerate(response_text):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_pos = i + 1
                                        break
                            if end_pos > 0:
                                result = json.loads(response_text[:end_pos])
                            else:
                                raise json_err
                        else:
                            raise json_err
                    except:
                        raise json_err
                else:
                    # Handle invalid escape sequences
                    print(f"  [WARNING] JSON decode error: {json_err}")
                    print(f"  [INFO] Attempting to fix escape sequences...")
                    # Replace invalid escape sequences with safe versions
                    # Fix common invalid escape patterns
                    response_text = response_text.replace('\\u', '\\\\u')  # Escape backslash before u
                    response_text = re.sub(r'\\([^"\\/bfnrtu])', r'\1', response_text)  # Remove invalid escapes
                    try:
                        result = json.loads(response_text)
                        print(f"  [INFO] Successfully parsed after fixing escape sequences")
                    except json.JSONDecodeError as json_err2:
                        raise json_err2

            # Ensure result is a list
            if isinstance(result, list):
                final_speeches_dict = result
            elif "speeches" in result:
                final_speeches_dict = result["speeches"]
            else:
                # Return original if parsing fails
                return validated_speeches

            # Validate and ensure input_type matches input_id
            # Also preserve original_speech from classified_speeches input
            validated_final_speeches = []
            for i, speech_dict in enumerate(final_speeches_dict):
                try:
                    input_id = speech_dict.get('input_id', 0)
                    speech_dict['input_type'] = INPUT_ID_MAPPING.get(input_id, "Unknown")
                    
                    # Set has_petition based on input_id if not already set
                    if 'has_petition' not in speech_dict:
                        speech_dict['has_petition'] = 'yes' if input_id == 11 else 'no'
                    # Set petition_count: default to 1 if petition but no count provided, 0 otherwise
                    if 'petition_count' not in speech_dict:
                        speech_dict['petition_count'] = 1 if input_id == 11 else 0
                    elif input_id != 11:
                        speech_dict['petition_count'] = 0
                    # has_expunged should be set by LLM based on content detection
                    # Only set as fallback if input_id is 13 (entire speech expunged) and not already set
                    if 'has_expunged' not in speech_dict:
                        speech_dict['has_expunged'] = 'yes' if input_id == 13 else 'no'
                    # If input_id is 13, ensure has_expunged is "yes" (entire speech is expunged)
                    elif input_id == 13 and speech_dict.get('has_expunged') != 'yes':
                        speech_dict['has_expunged'] = 'yes'

                    if 'expunged_count' not in speech_dict:
                        speech_dict['expunged_count'] = (
                            1 if speech_dict.get('has_expunged') == 'yes' or input_id == 13 else 0
                        )
                    elif speech_dict.get('has_expunged') == 'no':
                        speech_dict['expunged_count'] = 0
                    elif int(speech_dict.get('expunged_count', 0) or 0) < 1:
                        speech_dict['expunged_count'] = 1
                    
                    # Preserve original_speech and other fields from input by index
                    if i < len(validated_speeches):
                        original = validated_speeches[i]
                        speech_dict['original_speech'] = original.original_speech
                        speech_dict['speech_start_time'] = original.speech_start_time or ''
                        speech_dict['speech_start_time_english'] = original.speech_start_time_english or ''
                        # Preserve structure_flags and theme if not in response
                        if 'structure_flags' not in speech_dict:
                            speech_dict['structure_flags'] = original.structure_flags.model_dump()
                        if 'theme' not in speech_dict:
                            speech_dict['theme'] = original.theme.model_dump()
                    
                    # Validate and create ClassifiedSpeech model
                    validated_speech = ClassifiedSpeech(**speech_dict)
                    validated_final_speeches.append(validated_speech)
                except Exception as e:
                    print(f"  [WARNING] Failed to validate speech {i}: {e}")
                    # Try to use original speech if validation fails
                    if i < len(validated_speeches):
                        validated_final_speeches.append(validated_speeches[i])

            return validated_final_speeches
            
        except Exception as e:
            error_msg = str(e)
            # Show full error message (limit to 200 chars for readability)
            display_msg = error_msg if len(error_msg) <= 200 else error_msg[:200] + "..."
            print(f"  ⚠️ Attempt {attempt + 1}/{retry_attempts} failed: {display_msg}")
            if attempt < retry_attempts - 1:
                print(f"  ⏳ Retrying in 60s...")
                await asyncio.sleep(60)
            else:
                print(f"  ❌ All retry attempts failed. Full error: {error_msg}")
                return validated_speeches  # Return original on error

# Helper function to combine speeches spanning across pages if speaker names are similar
def combine_speeches_across_pages(all_speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine speeches that span across consecutive pages if speaker names are similar.
    Uses the speech with highest word count for input_id and categories.

    For AUDITOR-GENERAL'S REPORT / PAPERS PRESENTED: do not combine two chunks
    that share the same starting page; combining across a page step (e.g. 43→44)
    is still allowed.
    """
    if not all_speeches or len(all_speeches) <= 1:
        return all_speeches
    
    # Calculate word counts for each speech
    for speech in all_speeches:
        speech_text = speech.get('speech', '') or speech.get('original_speech', '')
        speech['word_count'] = len(speech_text.split()) if speech_text else 0
    
    # Sort by page number to process in order
    sorted_speeches = sorted(all_speeches, key=lambda s: s.get('page', 0))
    
    def are_speaker_names_similar(name1: str, name2: str) -> bool:
        """Simple check if speaker names are similar (same person)"""
        if not name1 or not name2:
            return False
        
        name1_clean = name1.strip().lower()
        name2_clean = name2.strip().lower()
        
        # Exact match
        if name1_clean == name2_clean:
            return True
        
        # Check if one name contains the other (for cases like "Gnanamuththu Srineshan" vs "Gnanamuthu Srineshan")
        # Extract key words (remove common titles and punctuation)
        def extract_key_words(name):
            # Remove common titles
            titles = ['the hon.', 'hon.', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'minister', 'deputy', 'speaker']
            words = name.split()
            key_words = [w for w in words if not any(title in w.lower() for title in titles)]
            return [w.lower().strip('.,()') for w in key_words if len(w) > 2]
        
        words1 = extract_key_words(name1)
        words2 = extract_key_words(name2)
        
        if not words1 or not words2:
            return False
        
        # Check if there's significant overlap in key words
        # If at least 2 key words match, consider them similar
        common_words = set(words1) & set(words2)
        if len(common_words) >= min(2, min(len(words1), len(words2))):
            return True
        
        # Check for substring matches (handles typos like "Gnanamuththu" vs "Gnanamuthu")
        if len(name1_clean) > 5 and len(name2_clean) > 5:
            # Check if one is a substring of the other (with some tolerance)
            if name1_clean in name2_clean or name2_clean in name1_clean:
                return True
            # Check similarity ratio (simple Levenshtein-like check)
            if abs(len(name1_clean) - len(name2_clean)) <= 2:
                # Check character overlap
                chars1 = set(name1_clean.replace(' ', ''))
                chars2 = set(name2_clean.replace(' ', ''))
                overlap = len(chars1 & chars2) / max(len(chars1), len(chars2)) if chars1 or chars2 else 0
                if overlap > 0.7:  # 70% character overlap
                    return True
        
        return False
    
    combined_speeches = []
    i = 0
    
    while i < len(sorted_speeches):
        current_speech = sorted_speeches[i].copy()
        current_page = current_speech.get('page', 0)
        current_speaker = current_speech.get('speaker_name', '').strip()
        
        # Look ahead to find speeches on consecutive pages with similar speaker names
        speeches_to_combine = [current_speech]
        j = i + 1
        
        while j < len(sorted_speeches):
            next_speech = sorted_speeches[j]
            next_page = next_speech.get('page', 0)
            next_speaker = next_speech.get('speaker_name', '').strip()
            
            # Check if pages are consecutive (within 1 page difference)
            if next_page - current_page > 1:
                break  # Gap too large, stop looking
            
            # Skip merge for certain headings
            NO_MERGE = {"AUDITOR GENERAL'S REPORT", "AUDITOR-GENERAL'S REPORT", "PAPERS PRESENTED"}
            def _blocked(s):
                for src in [s.get('structure_flags', {}).get('heading_text', []), s.get('heading_text', [])]:
                    if not src:
                        continue
                    for h in (src if isinstance(src, list) else [src]):
                        if str(h).strip().upper() in NO_MERGE:
                            return True
                return False
            blocked_here = _blocked(current_speech) or _blocked(next_speech)
            if blocked_here and (next_page == current_page):
                break

            # Check if speaker names are similar
            if current_speaker and next_speaker and are_speaker_names_similar(current_speaker, next_speaker):
                speeches_to_combine.append(next_speech.copy())
                current_page = next_page  # Update for next iteration
                j += 1
            else:
                break  # Different speaker, stop combining
        
        # If we have multiple speeches to combine
        if len(speeches_to_combine) > 1:
            # Find the speech with highest word count
            max_word_speech = max(speeches_to_combine, key=lambda s: s.get('word_count', 0))
            
            # Use categories from the speech with highest word count
            combined = max_word_speech.copy()
            
            # Merge speech text
            all_speech_texts = [s.get('speech', '') or s.get('original_speech', '') for s in speeches_to_combine]
            combined['speech'] = ' '.join(filter(None, all_speech_texts))
            
            # Merge original speech if available
            all_original_texts = [s.get('original_speech', '') for s in speeches_to_combine if s.get('original_speech')]
            if all_original_texts:
                combined['original_speech'] = ' '.join(all_original_texts)
            
            # Sum word counts
            combined['word_count'] = sum(s.get('word_count', 0) for s in speeches_to_combine)
            combined['aggregated_word_count'] = combined['word_count']
            combined['original_speech_word_count'] = len(combined.get('original_speech', '').split())
            
            # Collect all pages
            pages = sorted(list(set(s.get('page', 0) for s in speeches_to_combine)))
            combined['pages'] = pages
            combined['page'] = pages[0]  # Keep first page for compatibility
            
            # Use speaker name from highest word count speech
            combined['speaker_name'] = max_word_speech.get('speaker_name', '')
            
            # Add metadata
            combined['is_aggregated'] = True
            combined['aggregated_count'] = len(speeches_to_combine)

            # Preserve expunged flag: if ANY part is expunged, mark combined as expunged
            if any(s.get('has_expunged') == 'yes' for s in speeches_to_combine):
                combined['has_expunged'] = 'yes'
            combined['expunged_count'] = sum(
                int(s.get('expunged_count', 0) or 0) for s in speeches_to_combine
            )
            combined['speech_start_time'] = ''
            combined['speech_start_time_english'] = ''
            for s in speeches_to_combine:
                t = (s.get('speech_start_time') or '').strip()
                if t:
                    combined['speech_start_time'] = t
                    break
            for s in speeches_to_combine:
                te = (s.get('speech_start_time_english') or '').strip()
                if te:
                    combined['speech_start_time_english'] = te
                    break
            
            combined_speeches.append(combined)
            i = j  # Skip the combined speeches
        else:
            # Single speech, no combination needed
            combined_speeches.append(current_speech)
            i += 1
    
    print(f"  [OK] Combined {len(all_speeches)} speeches into {len(combined_speeches)} entries")
    
    return combined_speeches

# Helper function to normalize heading_text to always be a list
def normalize_heading_text(heading_text: Any) -> List[str]:
    """
    Normalize heading_text to always be a list.
    Handles cases where it might be a string, string representation of list, or already a list.
    """
    if heading_text is None:
        return []
    
    # If already a list, return it
    if isinstance(heading_text, list):
        return heading_text
    
    # If it's a string, try to parse it
    if isinstance(heading_text, str):
        heading_text = heading_text.strip()
        if not heading_text:
            return []
        
        # If it looks like a string representation of a list (starts with [ and ends with ])
        if heading_text.startswith('[') and heading_text.endswith(']'):
            try:
                # Try to parse it as JSON
                parsed = json.loads(heading_text)
                if isinstance(parsed, list):
                    return parsed
                else:
                    # If it's a single item, wrap it in a list
                    return [str(parsed)]
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, treat the whole string as a single heading
                # Remove brackets and return as list
                cleaned = heading_text.strip('[]').strip()
                return [cleaned] if cleaned else []
        else:
            # Single string, return as list
            return [heading_text]
    
    # For any other type, convert to string and wrap in list
    return [str(heading_text)] if heading_text else []

# Helper functions for robust taxonomy normalization/matching
def _normalize_taxonomy_text(value: Any) -> str:
    """Normalize taxonomy text for fuzzy-safe matching."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_main_category(category_value: Any) -> tuple[int, str]:
    """
    Resolve main category from flexible model outputs.
    Supports:
    - raw ID (e.g., 12)
    - prefixed labels (e.g., "12. Governance, Administration and Parliamentary Affairs")
    - canonical or near-canonical names (case/punctuation variations)
    """
    raw = "" if category_value is None else str(category_value).strip()
    if not raw:
        return 0, ""

    # Direct numeric category ID
    if str(raw).isdigit():
        cat_id = int(raw)
        if cat_id in MAIN_CATEGORY_MAPPING:
            return cat_id, MAIN_CATEGORY_MAPPING[cat_id]

    # Prefix pattern: "12. Name" / "12) Name" / "12 - Name"
    prefix_match = re.match(r"^\s*(\d{1,2})\s*[\.\)\-:]\s*(.+)$", raw)
    if prefix_match:
        cat_id = int(prefix_match.group(1))
        if cat_id in MAIN_CATEGORY_MAPPING:
            return cat_id, MAIN_CATEGORY_MAPPING[cat_id]

    # Exact (case-insensitive) match
    raw_lower = raw.lower()
    for cat_id, name in MAIN_CATEGORY_MAPPING.items():
        if raw_lower == name.lower():
            return cat_id, name

    # Normalized match (handles punctuation, "&"/"and", spacing differences)
    raw_norm = _normalize_taxonomy_text(raw)
    if raw_norm:
        for cat_id, name in MAIN_CATEGORY_MAPPING.items():
            if raw_norm == _normalize_taxonomy_text(name):
                return cat_id, name

        # Last-resort contains match (best effort, conservative)
        for cat_id, name in MAIN_CATEGORY_MAPPING.items():
            norm_name = _normalize_taxonomy_text(name)
            if norm_name and (norm_name in raw_norm or raw_norm in norm_name):
                return cat_id, name

    return 0, ""


def resolve_subcategory(subcategory_value: Any, main_category_id: int = 0) -> tuple[str, str]:
    """
    Resolve subcategory from flexible model outputs.
    Supports:
    - raw code (e.g., "12e")
    - formatted code (e.g., "12.e", "12 e")
    - prefixed labels (e.g., "12e Parliamentary Affairs", "e. Parliamentary Affairs")
    - canonical or near-canonical names (case/punctuation variations)
    """
    raw = "" if subcategory_value is None else str(subcategory_value).strip()
    if not raw:
        return "", ""

    raw_lower = raw.lower().strip()

    # Direct code: "12e"
    if raw_lower in SUB_CATEGORY_MAPPING:
        return raw_lower, SUB_CATEGORY_MAPPING[raw_lower]

    # Code with separators: "12.e", "12 e"
    code_match = re.match(r"^\s*(\d{1,2})\s*[\.\-\s]?\s*([a-z])\s*$", raw_lower)
    if code_match:
        code = f"{int(code_match.group(1))}{code_match.group(2)}"
        if code in SUB_CATEGORY_MAPPING:
            return code, SUB_CATEGORY_MAPPING[code]

    # Prefixed code with name: "12e Parliamentary Affairs"
    prefixed_code_match = re.match(r"^\s*(\d{1,2}[a-z])\b", raw_lower)
    if prefixed_code_match:
        code = prefixed_code_match.group(1)
        if code in SUB_CATEGORY_MAPPING:
            return code, SUB_CATEGORY_MAPPING[code]

    # Letter-only prefix with known main category: "e. Parliamentary Affairs"
    letter_match = re.match(r"^\s*([a-z])\s*[\.\)\-:]", raw_lower)
    if letter_match and main_category_id:
        candidate_code = f"{main_category_id}{letter_match.group(1)}"
        if candidate_code in SUB_CATEGORY_MAPPING:
            return candidate_code, SUB_CATEGORY_MAPPING[candidate_code]

    # Name-based exact/normalized match (prefer same main category when available)
    candidate_items = list(SUB_CATEGORY_MAPPING.items())
    if main_category_id:
        prefix = f"{main_category_id}"
        preferred = [(k, v) for k, v in candidate_items if k.startswith(prefix)]
        if preferred:
            candidate_items = preferred + [(k, v) for k, v in candidate_items if not k.startswith(prefix)]

    for code, name in candidate_items:
        if raw_lower == name.lower():
            return code, name

    raw_norm = _normalize_taxonomy_text(raw)
    if raw_norm:
        for code, name in candidate_items:
            if raw_norm == _normalize_taxonomy_text(name):
                return code, name
        for code, name in candidate_items:
            norm_name = _normalize_taxonomy_text(name)
            if norm_name and (norm_name in raw_norm or raw_norm in norm_name):
                return code, name

    return "", ""

# Helper functions to get category mappings
def get_main_category_id(category_name: str) -> int:
    """Get the ID for a main category name"""
    cat_id, _ = resolve_main_category(category_name)
    return cat_id

def get_subcategory_code(subcategory_name: str) -> str:
    """Get the code for a subcategory name"""
    code, _ = resolve_subcategory(subcategory_name)
    return code

_ADJOURNMENT_MAIN_HEADING_RE = re.compile(
    r"^(ADJOURNMENTS?|ADJOURNMENT\s+MOTION|MOTION\s+AT\s+ADJOURNMENT\s+TIME)\s*$",
    re.IGNORECASE,
)


def _is_bare_adjournment_section_heading(heading: str) -> bool:
    """True if this is only the main adjournment section title (sub-topic comes next, maybe next page)."""
    if not heading or not str(heading).strip():
        return False
    return bool(_ADJOURNMENT_MAIN_HEADING_RE.match(str(heading).strip()))


def apply_adjournment_subheading_prefix_to_raw_speeches(
    raw_speeches: List[Dict[str, Any]],
    pending: bool,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    After extraction, the model often misses the trilingual sub-heading when it starts on a new page.
    When we see a bare ADJOURNMENT section title, prepend ADJOURNMENTS to the very next non-empty
    heading only (same page or next). Carries pending across pages via the bool returned.
    """
    prefix = "ADJOURNMENTS "
    for item in raw_speeches:
        h = item.get("heading")
        if h is None:
            continue
        hs = str(h).strip()
        if not hs:
            continue
        if pending:
            if not hs.upper().startswith("ADJOURNMENTS "):
                item["heading"] = prefix + hs
            pending = False
            continue
        if _is_bare_adjournment_section_heading(hs):
            pending = True
    return raw_speeches, pending


# Helper function to fill missing topics from previous speech
def fill_missing_topics(speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fill missing topics by copying from the previous speech.
    If a speech has no theme or empty main_category, copy from the speech above it.
    """
    if not speeches:
        return speeches

    last_valid_theme = None

    for speech in speeches:
        theme = speech.get('theme', {})

        # Check if theme is missing or empty
        if not theme or not theme.get('main_category') or theme.get('main_category', '').strip() == '':
            if last_valid_theme:
                # Copy the last valid theme
                speech['theme'] = last_valid_theme.copy()
        else:
            # Store this as the last valid theme
            last_valid_theme = theme.copy()

    return speeches

# Helper function to fill missing headings from previous speech
def fill_missing_headings(speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fill missing headings by copying from the previous speech.
    If a speech has no heading_text or empty heading_text, copy from the speech above it.
    This is crucial for input ID classification which relies on section headings.
    """
    if not speeches:
        return speeches

    last_valid_heading = None

    for speech in speeches:
        structure_flags = speech.get('structure_flags', {})
        heading_text = structure_flags.get('heading_text', [])

        # Check if heading is missing or empty
        if not heading_text or (isinstance(heading_text, list) and len(heading_text) == 0):
            if last_valid_heading:
                # Copy the last valid heading
                if 'structure_flags' not in speech:
                    speech['structure_flags'] = {}
                speech['structure_flags']['heading_text'] = last_valid_heading.copy()
                speech['structure_flags']['has_headings'] = True
        else:
            # Store this as the last valid heading (only if it's not empty)
            if isinstance(heading_text, list) and len(heading_text) > 0:
                last_valid_heading = heading_text.copy()

    return speeches


def strip_leading_bracket_tags(speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove square-bracketed content at the very start of speech and original_speech
    fields (e.g. '[අ.ශා. 12.19]'). These are Sinhala page/time markers injected by OCR
    and should not count toward word counts or scores.

    Skip stripping when any heading on the speech contains "adjournment" (case-insensitive),
    so adjournment-debate bracketed cues at the start of the text are preserved.
    """
    def _heading_text_fragments(speech: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        sf = speech.get('structure_flags') or {}
        if isinstance(sf, dict):
            ht = sf.get('heading_text')
            if isinstance(ht, list):
                out.extend(str(h) for h in ht)
            elif ht is not None and str(ht).strip():
                out.append(str(ht))
        ht2 = speech.get('heading_text')
        if isinstance(ht2, list):
            out.extend(str(h) for h in ht2)
        elif ht2 is not None and str(ht2).strip():
            out.append(str(ht2))
        return out

    def _heading_has_adjournment(speech: Dict[str, Any]) -> bool:
        return any('adjournment' in frag.lower() for frag in _heading_text_fragments(speech))

    bracket_re = re.compile(r'^\s*\[[^\]]*\]\s*')
    stripped = 0
    for s in speeches:
        if _heading_has_adjournment(s):
            continue
        for field in ('speech', 'original_speech'):
            val = s.get(field, '')
            if val and bracket_re.match(val):
                s[field] = bracket_re.sub('', val)
                stripped += 1
    if stripped:
        print(f"  [OK] Stripped leading bracket tags from {stripped} speech field(s)")
    return speeches


# Helper function to fill missing speaker names from previous speech
def fill_missing_speakers(speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fill missing speaker names by copying from the previous speech.
    If a speech has no speaker_name or empty speaker_name (null from extraction),
    copy from the speech above it. This handles page breaks where speeches continue
    without repeating the speaker name.
    """
    if not speeches:
        return speeches

    last_valid_speaker = None

    for speech in speeches:
        speaker_name = speech.get('speaker_name', '')

        # Check if speaker is missing, empty, or null
        if not speaker_name or speaker_name == '' or speaker_name is None:
            if last_valid_speaker:
                # Copy the last valid speaker
                speech['speaker_name'] = last_valid_speaker
        else:
            # Store this as the last valid speaker
            last_valid_speaker = speaker_name

    return speeches


# Helper function to merge continuation speeches across pages
def merge_continuation_speeches(speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge speeches that are continuations across pages.
    If consecutive speeches have:
    - Same speaker
    - Same heading
    - Consecutive or same page numbers
    Then merge them into a single speech.

    For AUDITOR-GENERAL'S REPORT / PAPERS PRESENTED: do not merge two segments
    on the same page (keeps each item separate); merging across a page boundary
    is still allowed (continuation onto next page).

    Metadata (theme, input_id, input_type, speaker_name, structure_flags, etc.)
    is taken from the segment with the highest word count. Speech texts and
    original_speech texts are concatenated in order.
    """
    if not speeches or len(speeches) <= 1:
        return speeches

    NO_MERGE_HEADINGS = {"AUDITOR GENERAL'S REPORT", "AUDITOR-GENERAL'S REPORT", "PAPERS PRESENTED"}

    def _heading_blocks_merge(speech_dict: Dict[str, Any]) -> bool:
        """Check both structure_flags.heading_text and top-level heading_text."""
        sources = [
            speech_dict.get('structure_flags', {}).get('heading_text', []),
            speech_dict.get('heading_text', []),
        ]
        for heading_list in sources:
            if not heading_list:
                continue
            for h in (heading_list if isinstance(heading_list, list) else [heading_list]):
                if str(h).strip().upper() in NO_MERGE_HEADINGS:
                    return True
        return False

    def _word_count(s: Dict[str, Any]) -> int:
        txt = (s.get('original_speech') or s.get('speech') or '').strip()
        return len(txt.split()) if txt else 0

    def _merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = max(group, key=_word_count)
        merged = best.copy()

        all_speech = [s.get('speech', '') for s in group]
        merged['speech'] = ' '.join(filter(None, all_speech))

        all_orig = [s.get('original_speech', '') for s in group if s.get('original_speech')]
        if all_orig:
            merged['original_speech'] = ' '.join(all_orig)

        total_wc = sum(_word_count(s) for s in group)
        merged['word_count'] = total_wc
        merged['aggregated_word_count'] = total_wc
        merged['original_speech_word_count'] = len(merged.get('original_speech', '').split())

        pages = sorted({s.get('page', 0) for s in group})
        merged['pages'] = pages
        merged['page'] = pages[0]

        merged['speaker_name'] = best.get('speaker_name', '')
        merged['is_aggregated'] = True
        merged['aggregated_count'] = len(group)

        if any(s.get('has_expunged') == 'yes' for s in group):
            merged['has_expunged'] = 'yes'
        merged['expunged_count'] = sum(int(s.get('expunged_count', 0) or 0) for s in group)

        merged['speech_start_time'] = ''
        merged['speech_start_time_english'] = ''
        for s in group:
            t = (s.get('speech_start_time') or '').strip()
            if t:
                merged['speech_start_time'] = t
                break
        for s in group:
            te = (s.get('speech_start_time_english') or '').strip()
            if te:
                merged['speech_start_time_english'] = te
                break

        if any(s.get('has_petition') == 'yes' for s in group):
            merged['has_petition'] = 'yes'
            merged['petition_count'] = sum(s.get('petition_count', 0) for s in group)

        if any(s.get('structure_flags', {}).get('has_tables', False) for s in group):
            sf = merged.get('structure_flags', {})
            if isinstance(sf, dict):
                sf['has_tables'] = True
            elif hasattr(sf, 'has_tables'):
                sf.has_tables = True
            merged['structure_flags'] = sf

        return merged

    merged_speeches = []
    group: List[Dict[str, Any]] = [speeches[0].copy()]

    for speech in speeches[1:]:
        prev = group[-1]

        same_speaker = (prev.get('speaker_name') == speech.get('speaker_name'))

        current_heading = prev.get('structure_flags', {}).get('heading_text', []) or prev.get('heading_text', [])
        next_heading = speech.get('structure_flags', {}).get('heading_text', []) or speech.get('heading_text', [])
        same_heading = (current_heading == next_heading)

        current_page = prev.get('page', 0)
        next_page = speech.get('page', 0)
        consecutive_pages = (next_page - current_page <= 1)

        blocked = _heading_blocks_merge(prev) or _heading_blocks_merge(speech)
        same_page = current_page == next_page
        blocked_same_page_only = blocked and same_page

        if same_speaker and same_heading and consecutive_pages and not blocked_same_page_only:
            group.append(speech.copy())
        else:
            merged_speeches.append(_merge_group(group) if len(group) > 1 else group[0])
            group = [speech.copy()]

    merged_speeches.append(_merge_group(group) if len(group) > 1 else group[0])

    return merged_speeches




# Excel hard limit per cell is 32,767. Stay well under; some CSV imports still spill near the limit.
EXCEL_MAX_CELL_CHARS = 30000
_CSV_SPEECH_TRUNC_SUFFIX = ' ...[truncated; full text in JSON]'


def _truncate_cell_for_csv(text: str, max_chars: int) -> tuple[str, int]:
    """Return (cell text, full character length). Long text is only the start + suffix."""
    raw = text or ''
    n = len(raw)
    if n <= max_chars:
        return raw, n
    suf = _CSV_SPEECH_TRUNC_SUFFIX
    keep = max(0, max_chars - len(suf))
    return raw[:keep] + suf, n


def _structured_export_fieldnames() -> List[str]:
    return [
        'page',
        'speaker_name',
        'speech_start_time',
        'speech_start_time_english',
        'original_speech',
        'original_speech_char_count',
        'original_speech_word_count',
        'original_speech_line_count',
        'input_score',
        'speech',
        'speech_char_count',
        'heading_text',
        'main_category',
        'main_category_id',
        'main_category_name',
        'subcategory',
        'subcategory_code',
        'subcategory_name',
        'justification_1',
        'justification_2',
        'input_id',
        'input_type',
        'has_tables',
        'has_petition',
        'petition_count',
        'has_expunged',
        'expunged_count',
        'was_answered',
    ]


def _classified_speech_to_export_row(
    item: Union[Dict[str, Any], ClassifiedSpeech],
) -> Dict[str, Any]:
    """One classified speech -> flat dict for CSV/XLSX (Excel-safe cell sizes)."""
    if isinstance(item, ClassifiedSpeech):
        item_dict = item.model_dump()
    else:
        item_dict = item

    theme = item_dict.get('theme', {})
    structure_flags = item_dict.get('structure_flags', {})

    heading_text = structure_flags.get('heading_text', [])
    if isinstance(heading_text, list):
        heading_text = '; '.join(heading_text)
    if len(heading_text) > EXCEL_MAX_CELL_CHARS:
        heading_text = heading_text[: EXCEL_MAX_CELL_CHARS - 35] + '...[truncated]'

    raw_main_category = theme.get('main_category', '')
    raw_subcategory = theme.get('subcategory', '')
    main_category_id, resolved_main_category = resolve_main_category(raw_main_category)
    subcategory_code, resolved_subcategory = resolve_subcategory(raw_subcategory, main_category_id)
    main_category = resolved_main_category or raw_main_category
    subcategory = resolved_subcategory or raw_subcategory

    original_speech = item_dict.get('original_speech', '')
    original_speech_word_count = len(original_speech.split()) if original_speech else 0
    original_speech_line_count = math.ceil(original_speech_word_count / 6.3) if original_speech_word_count else 0
    input_score = calculate_input_score(
        item_dict.get('input_id', ''),
        original_speech_line_count,
        item_dict.get('petition_count', 0),
    )

    speech_text = (item_dict.get('speech', '') or '').replace('\n', ' ').replace('\r', ' ')
    original_speech_clean = (original_speech or '').replace('\n', ' ').replace('\r', ' ')
    speaker_name_clean = (item_dict.get('speaker_name') or '').replace('\n', ' ').replace('\r', ' ')

    original_speech_cell, orig_char_count = _truncate_cell_for_csv(
        original_speech_clean, EXCEL_MAX_CELL_CHARS
    )
    speech_cell, speech_char_count = _truncate_cell_for_csv(speech_text, EXCEL_MAX_CELL_CHARS)

    has_petition = item_dict.get('has_petition', 'no')
    has_expunged = item_dict.get('has_expunged', 'no')
    if has_petition == 'no' and item_dict.get('input_id') == 11:
        has_petition = 'yes'
    if has_expunged == 'no' and item_dict.get('input_id') == 13:
        has_expunged = 'yes'
    expunged_count = int(item_dict.get('expunged_count', 0) or 0)
    if has_expunged == 'no':
        expunged_count = 0
    elif expunged_count < 1:
        expunged_count = 1

    justification_raw = (theme.get('justification', '') or '').replace('\n', ' ').replace('\r', ' ')
    j1 = justification_raw[:EXCEL_MAX_CELL_CHARS]
    j2 = justification_raw[EXCEL_MAX_CELL_CHARS : EXCEL_MAX_CELL_CHARS * 2]
    if len(justification_raw) > EXCEL_MAX_CELL_CHARS * 2:
        j2 = j2[: max(0, EXCEL_MAX_CELL_CHARS - 50)] + '...[truncated; see JSON]'

    speech_start_time = (item_dict.get('speech_start_time') or '').replace('\n', ' ').replace('\r', ' ')
    speech_start_time_english = (item_dict.get('speech_start_time_english') or '').replace('\n', ' ').replace('\r', ' ')

    return {
        'page': item_dict.get('page', ''),
        'speaker_name': speaker_name_clean,
        'speech_start_time': speech_start_time,
        'speech_start_time_english': speech_start_time_english,
        'original_speech': original_speech_cell,
        'original_speech_char_count': orig_char_count,
        'original_speech_word_count': original_speech_word_count,
        'original_speech_line_count': original_speech_line_count,
        'input_score': input_score,
        'speech': speech_cell,
        'speech_char_count': speech_char_count,
        'heading_text': heading_text,
        'main_category': main_category,
        'main_category_id': main_category_id,
        'main_category_name': MAIN_CATEGORY_MAPPING.get(main_category_id, ''),
        'subcategory': subcategory,
        'subcategory_code': subcategory_code,
        'subcategory_name': SUB_CATEGORY_MAPPING.get(subcategory_code, ''),
        'justification_1': j1,
        'justification_2': j2,
        'input_id': item_dict.get('input_id', ''),
        'input_type': item_dict.get('input_type', ''),
        'has_tables': structure_flags.get('has_tables', False),
        'has_petition': has_petition,
        'petition_count': item_dict.get('petition_count', 0),
        'has_expunged': has_expunged,
        'expunged_count': expunged_count,
        'was_answered': item_dict.get('was_answered', ''),
    }


def save_to_csv_with_structure(speeches: Union[List[Dict[str, Any]], List[ClassifiedSpeech]], csv_file: str):
    """
    Save classified speeches to CSV (UTF-8 BOM). Very long speech/original_speech is truncated
    to one cell (see EXCEL_MAX_CELL_CHARS); full text remains in the JSON output.
    """
    if not speeches:
        return

    fieldnames = _structured_export_fieldnames()
    with open(csv_file, 'w', newline='', encoding='utf-8-sig', errors='replace') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for item in speeches:
            writer.writerow(_classified_speech_to_export_row(item))


def save_to_xlsx_with_structure(speeches: Union[List[Dict[str, Any]], List[ClassifiedSpeech]], xlsx_file: str) -> bool:
    """Same columns as structured CSV as native .xlsx. Returns False if openpyxl is missing."""
    if not OPENPYXL_AVAILABLE or not speeches:
        return False

    fieldnames = _structured_export_fieldnames()
    wb = Workbook()
    ws = wb.active
    ws.title = 'classified'
    ws.append(fieldnames)
    for item in speeches:
        rowd = _classified_speech_to_export_row(item)
        ws.append([rowd.get(k, '') for k in fieldnames])
    wb.save(xlsx_file)
    return True
# ============================================================================
# PDF DOCUMENT PROCESSING
# ============================================================================

async def classify_hansard_pdf_async(pdf_path: str, output_file: str, max_concurrent: int = 1, dpi: int = 200, max_pages: int = None):
    """
    Process a Hansard PDF document sequentially, page by page.
    Sequential mode allows passing the last 5 classified speeches as context
    to the input ID classification agent for each new page.

    Args:
        pdf_path: Path to the PDF file
        output_file: Output JSON file path
        max_concurrent: Semaphore limit for individual API calls within a page (default 1)
        dpi: Resolution for PDF to image conversion (default 200)
        max_pages: If set, only process the first N pages (useful for testing)
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image is required for PDF processing. Run: pip install pdf2image")

    # Convert PDF to images
    print(f"\n[Step 1] Converting PDF to images...")
    page_images = convert_pdf_to_images(pdf_path, dpi=dpi)

    # Limit pages if max_pages is specified
    if max_pages is not None:
        page_images = page_images[:max_pages]
        print(f"  [INFO] Limiting to first {max_pages} pages for this run")

    print(f"\n[Step 2] Processing {len(page_images)} pages sequentially...")
    print(f"Pipeline per page: Extract → Translate → Classify → Input ID → Merge\n")

    semaphore = asyncio.Semaphore(max_concurrent)
    all_speeches = []
    adjournment_subheading_pending = False

    for i, page_data in enumerate(page_images, 1):
        page_num = page_data.get('metadata', {}).get('page', i)
        print(f"[{i}/{len(page_images)}] Page {page_num}")

        try:
            # Step 1: Extract speakers
            async with semaphore:
                print(f"  [Gemini] Extracting speakers...")
                raw_speeches = await async_speaker_extraction_agent(page_data)

            if not raw_speeches:
                print(f"  [SKIP] No speeches found")
                continue

            raw_speeches, adjournment_subheading_pending = (
                apply_adjournment_subheading_prefix_to_raw_speeches(
                    raw_speeches, adjournment_subheading_pending
                )
            )

            print(f"  Extracted {len(raw_speeches)} speech(es)")

            # Step 2: Translate and structure
            async with semaphore:
                print(f"  [Gemini] Translating...")
                translation_result = await async_translation_and_structure_agent(page_data, raw_speeches)

            extracted_speeches = translation_result.get("extracted_speeches", [])
            if not extracted_speeches:
                print(f"  [SKIP] Translation failed")
                continue

            print(f"  Translated {len(extracted_speeches)} speech(es)")

            # Attach original speeches by index; keep time stamp from raw if translation omits it
            for idx, speech in enumerate(extracted_speeches):
                speech['original_speech'] = raw_speeches[idx].get('speech', '') if idx < len(raw_speeches) else ''
                if idx < len(raw_speeches):
                    raw_t = (raw_speeches[idx].get('speech_start_time') or '').strip()
                    tr_t = (speech.get('speech_start_time') or '').strip()
                    speech['speech_start_time'] = raw_t or tr_t
                speech['speech_start_time_english'] = (speech.get('speech_start_time_english') or '').strip()

            # Python merges
            extracted_speeches = strip_leading_bracket_tags(extracted_speeches)
            extracted_speeches = fill_missing_speakers(extracted_speeches)
            extracted_speeches = merge_continuation_speeches(extracted_speeches)
            extracted_speeches = fill_missing_topics(extracted_speeches)
            extracted_speeches = fill_missing_headings(extracted_speeches)

            # Step 3: Classify themes
            async with semaphore:
                print(f"  [Gemini] Classifying themes...")
                classified_speeches = await async_content_classification_agent(page_data, extracted_speeches)

            if not classified_speeches:
                print(f"  [SKIP] Theme classification failed")
                continue

            print(f"  Classified {len(classified_speeches)} speech(es)")

            # Get context speeches from previous pages (last 5 speeches)
            context_speeches = all_speeches[-5:] if all_speeches else []

            # Step 4: Assign input IDs with cross-page context
            async with semaphore:
                print(f"  [Gemini] Assigning input IDs...")
                final_speeches = await async_input_id_classification_agent(page_data, classified_speeches, context_speeches)

            if final_speeches:
                print(f"  [OK] Processed {len(final_speeches)} speech(es)")
                for idx, speech in enumerate(final_speeches[:3], 1):
                    # Handle both ClassifiedSpeech and dict
                    if isinstance(speech, ClassifiedSpeech):
                        speaker = speech.speaker_name or "Unknown"
                        category = speech.theme.main_category if speech.theme else "N/A"
                        input_type = speech.input_type or "N/A"
                        speech_preview = (speech.speech or "")[:60].replace('\n', ' ')
                    else:
                        speaker = speech.get("speaker_name", "Unknown")
                        category = speech.get("theme", {}).get("main_category", "N/A")
                        input_type = speech.get("input_type", "N/A")
                        speech_preview = speech.get("speech", "")[:60].replace('\n', ' ')
                    print(f"    {idx}. {speaker} | {category} | {input_type} | {speech_preview}...")
                if len(final_speeches) > 3:
                    print(f"    ... and {len(final_speeches) - 3} more")
                # Store only dicts in all_speeches for downstream helpers
                all_speeches.extend([s.model_dump() if isinstance(s, ClassifiedSpeech) else s for s in final_speeches])
            else:
                print(f"  [SKIP] Input ID classification failed")

        except Exception as e:
            print(f"  [ERROR] {str(e)}\n")
            continue

    # Final post-processing across all pages
    print(f"\n[Step 3] Post-processing: merging continuations...")

    # Ensure all_speeches are plain dicts for post-processing helpers
    all_speeches = [s.model_dump() if isinstance(s, ClassifiedSpeech) else s for s in all_speeches]

    all_speeches = strip_leading_bracket_tags(all_speeches)
    all_speeches = fill_missing_speakers(all_speeches)
    all_speeches = merge_continuation_speeches(all_speeches)
    all_speeches = fill_missing_topics(all_speeches)
    all_speeches = fill_missing_headings(all_speeches)

    # Step 4: Combine speeches spanning across pages
    print(f"\n[Step 4] Combining speeches spanning across pages...")
    all_speeches = combine_speeches_across_pages(all_speeches)

    # Step 5: Remove short speeches (<5 computed lines), with allowlist exceptions
    print(f"\n[Step 5] Filtering short speeches...")
    all_speeches = filter_short_speeches(all_speeches, min_lines=5, keep_input_ids=KEEP_SHORT_SPEECH_INPUT_IDS)

    # Step 6: Re-merge consecutive speeches with similar speaker names
    print(f"\n[Step 6] Merging similar-speaker speeches after filtering...")
    all_speeches = merge_similar_speaker_speeches(all_speeches)

    # Step 7: Mark whether questions were answered
    print(f"\n[Step 7] Marking answered questions...")
    all_speeches = mark_answered_questions(all_speeches)

    # Step 8: Remove address rows (input_id 30)
    print(f"\n[Step 8] Removing address rows (input_id 30)...")
    all_speeches = remove_address_rows(all_speeches)

    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_speeches, f, indent=2, ensure_ascii=False)

    # Save CSV + optional XLSX (open Excel with .xlsx — not .json)
    csv_file = output_file.replace('.json', '.csv')
    save_to_csv_with_structure(all_speeches, csv_file)
    xlsx_file = output_file.replace('.json', '.xlsx')
    if save_to_xlsx_with_structure(all_speeches, xlsx_file):
        print(f"[SUCCESS] Also saved to {xlsx_file}")
    else:
        print(f"[INFO] Skipped {xlsx_file} (install openpyxl for native Excel export)")

    print(f"\n[SUCCESS] Processed {len(page_images)} pages from PDF")
    print(f"[SUCCESS] Extracted {len(all_speeches)} speeches/items")
    print(f"[SUCCESS] Saved to {output_file}")
    print(f"[SUCCESS] Also saved to {csv_file}")

    # Summary statistics
    if all_speeches:
        categories = {}
        for speech in all_speeches:
            cat = speech.get("theme", {}).get("main_category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1

        print("\nCategory breakdown:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")

    print_cost_summary()

    return all_speeches


def classify_hansard_pdf(pdf_path: str, output_file: str, dpi: int = 200, max_pages: int = None):
    """
    Synchronous wrapper for PDF processing. Processes pages one by one.
    """
    return asyncio.run(classify_hansard_pdf_async(pdf_path, output_file, max_concurrent=1, dpi=dpi, max_pages=max_pages))


if __name__ == "__main__":
    import sys

    # ─── Process novhansardtest2.pdf ───────────────────
    TEST_PDF      = "9thoctHansard.pdf"
    TEST_OUTPUT   = "9thoctHansard_classified.json"
    TEST_PAGES    = None  # Process all pages
    TEST_DPI      = 200

    print(f"Hansard 4-Agent Sequential Classifier (Trilingual)")
    print(f"====================================================")
    print(f"Input:     {TEST_PDF}")
    print(f"Output:    {TEST_OUTPUT}")
    print(f"Mode:      Sequential (context-aware, all pages)")
    print(f"PDF DPI:   {TEST_DPI}")
    print(f"")
    print(f"Pipeline (per page):")
    print(f"  1)   Speaker Extraction         [Gemini Vision - gemini-3-flash-preview]")
    print(f"  2)   Translation + Structure    [Gemini Flash  - gemini-3-flash-preview]")
    print(f"  3)   Theme Classification       [Gemini Flash  - gemini-3-flash-preview]")
    print(f"  4)   Input ID Classification    [Gemini Flash  - gemini-3-flash-preview]")
    print(f"  5)   Interruption Merge         [Gemini Flash  - gemini-3-flash-preview]")
    print(f"")
    # Avoid UnicodeEncodeError on Windows consoles with non-UTF8 codepages
    print("Process: Extract -> Translate -> Classify -> Input ID -> Merge -> Aggregate")
    print(f"Monitoring: LangSmith")
    print(f"Languages:  Sinhala, Sri Lankan Tamil, English\n")

    classify_hansard_pdf(
        pdf_path    = TEST_PDF,
        output_file = TEST_OUTPUT,
        dpi         = TEST_DPI,
        max_pages   = TEST_PAGES,
    )
 