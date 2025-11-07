import argparse
import json
import os
import sys
import time
from typing import Any, Dict


def _ensure_openai_available() -> None:
    
    try:
        
        from openai import OpenAI  
    except Exception as exc:  
        msg = (
            "Missing dependency: openai.\n"
            "Install it with: pip install openai\n"
            f"Underlying import error: {exc}"
        )
        print(msg, file=sys.stderr)
        sys.exit(2)


def _extract_pdf_text(pdf_path: str) -> str:
    
    
    try:
        
        try:
            import warnings  
            from cryptography.utils import CryptographyDeprecationWarning  
            warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
        except Exception:
            pass
        try:
            from pypdf import PdfReader  
        except Exception:  
            from PyPDF2 import PdfReader  

        reader = PdfReader(pdf_path)
        pages = []
        for p in getattr(reader, "pages", []):
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        if text and text.strip():
            return text
    except Exception:
        pass

    
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  

        text = pdfminer_extract_text(pdf_path) or ""
        if text and text.strip():
            return text
    except Exception:
        pass

    raise RuntimeError(
        "Failed to extract text from PDF. Install one of: 'pip install pypdf' or 'pip install pdfminer.six'."
    )


def _wait_for_file_active(*args, **kwargs):  
    raise NotImplementedError("File API path removed: using local PDF text parsing instead.")


def _build_openai_json_schema() -> Dict[str, Any]:
    """Return a JSON Schema enforcing the expected output shape.

    This schema matches the requested shape where projects/internships/achievements
    are arrays of single-key objects: [{"Title": "Description"}].
    """
    single_kv_obj: Dict[str, Any] = {
        "type": "object",
        "minProperties": 1,
        "maxProperties": 1,
        "additionalProperties": {"type": "string"},
    }

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "cgpa": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "certs": {"type": "array", "items": {"type": "string"}},
            "projects": {"type": "array", "items": single_kv_obj},
            "internships": {"type": "array", "items": single_kv_obj},
            "achievements": {"type": "array", "items": single_kv_obj},
        },
        "required": [
            "cgpa",
            "skills",
            "certs",
            "projects",
            "internships",
            "achievements",
        ],
        "additionalProperties": False,
    }
    return schema


def _list_openai_models(api_key: str | None) -> list[str]:
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        resp = client.models.list()
        
        names = sorted(m.id for m in getattr(resp, "data", []) if isinstance(getattr(m, "id", None), str) and "gpt" in m.id)
        return names
    except Exception:
        return []


def _select_model_name(preferred: str | None) -> str:
    """Return requested model or default to OpenAI gpt-4.1-mini."""
    
    if preferred:
        normalized = preferred.strip().lower().replace(" ", "-")
        return normalized
    return "gpt-4.1-mini"


EXTRACTION_PROMPT = (
    "You are an information extraction system. You will be given resume text.\n"
    "Extract the following fields and return ONLY valid JSON with this EXACT shape:\n\n"
    "{\n"
    "  \"cgpa\": \"string\",\n"
    "  \"skills\": [\"string\"],\n"
    "  \"certs\": [\"string\"],\n"
    "  \"projects\": [ { \"<Project Title>\": \"<One-sentence description>\" } ],\n"
    "  \"internships\": [ { \"<Role Org (Dates)>\": \"<One-sentence description>\" } ],\n"
    "  \"achievements\": [ { \"<Title>\": \"<One-sentence description>\" } ]\n"
    "}\n\n"
    "Rules:\n"
    "- Do not invent or hallucinate content.\n"
    "- If a field is missing, use an empty string for scalars or an empty array for lists.\n"
    "- For cgpa: extract numeric value if present (e.g., '9.27'). Prefer CGPA/GPA over percentage.\n"
    "- For skills: return a flat, deduplicated list of skill phrases.\n"
    "- For certs: each item is one line like 'Name  Organization (Dates)'.\n"
    "- For projects/internships/achievements: each array item must be an object with exactly one key (the title) and a short description value.\n"
)


def extract_resume(
    pdf_path: str,
    *,
    api_key: str | None = None,
    model_name: str = "gpt-4.1-mini",
    wait_timeout_sec: int = 120,
) -> Dict[str, Any]:
    """Extract resume data from a PDF via OpenAI and return a dict.

    Returns a dictionary with keys: cgpa, skills, certs, projects, internships, achievements.
    Uses local PDF-to-text; no file upload is used.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Input must be a .pdf file")

    # Extract text locally
    resume_text = _extract_pdf_text(pdf_path)
    if not resume_text.strip():
        raise ValueError("No text extracted from PDF (non-text or scanned PDF without OCR).")

    _ensure_openai_available()
    from openai import OpenAI

    api_key_effective = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key_effective:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Provide --api-key or set OPENAI_API_KEY env var."
        )

    
    selected_model = _select_model_name(model_name)
    client = OpenAI(api_key=api_key_effective)

    prompt = (
        f"{EXTRACTION_PROMPT}\n\n"
        f"Resume Text (between <<RESUME>> and <<END>>):\n"
        f"<<RESUME>>\n{resume_text}\n<<END>>\n"
    )

    schema = _build_openai_json_schema()

    
    raw_text = None
    try:
        response = client.responses.create(
            model=selected_model,
            input=prompt,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "resume_schema",
                    "schema": schema,
                    
                    "strict": False,
                },
            },
        )
        raw_text = getattr(response, "output_text", None)
        if not raw_text:
            
            try:
                parts: list[str] = []
                for item in getattr(response, "output", []) or []:
                    for piece in getattr(item, "content", []) or []:
                        t = getattr(piece, "text", None)
                        if isinstance(t, str):
                            parts.append(t)
                raw_text = "".join(parts) if parts else None
            except Exception:
                raw_text = None
    except Exception:
        raw_text = None

    if not raw_text:
       
        try:
            chat = client.chat.completions.create(
                model=selected_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract structured resume data and reply with JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw_text = (
                getattr(chat, "choices", [])[0]
                .message
                .content
                if getattr(chat, "choices", None)
                else None
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    if not isinstance(raw_text, str):
        raw_text = "{}"

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw_text[start : end + 1])
            else:
                raise
        except Exception:
            raise ValueError(f"Model did not return valid JSON: {exc}\nRaw: {raw_text[:500]}") from exc

    
    def _get(k, default):
        return data.get(k, default) if isinstance(data, dict) else default

    result: Dict[str, Any] = {
        "cgpa": str(_get("cgpa", "")) if _get("cgpa", "") is not None else "",
        "skills": _get("skills", []) or [],
        "certs": _get("certs", []) or [],
        "projects": _get("projects", []) or [],
        "internships": _get("internships", []) or [],
        "achievements": _get("achievements", []) or [],
    }

    
    for key in ("skills", "certs", "projects", "internships", "achievements"):
        if not isinstance(result[key], list):
            result[key] = []

    
    def _to_single_kv_list(items: list[Any]) -> list[Dict[str, str]]:
        norm: list[Dict[str, str]] = []
        for it in items:
            if isinstance(it, dict):
                
                if set(it.keys()) >= {"title", "description"}:
                    title = str(it.get("title", "")).strip()
                    desc = str(it.get("description", "")).strip()
                    if title:
                        norm.append({title: desc})
                else:
                    
                    if len(it) == 1:
                        k, v = next(iter(it.items()))
                        norm.append({str(k): str(v)})
                    else:
                       
                        for k, v in it.items():
                            if isinstance(k, str):
                                norm.append({k: str(v)})
                                break
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                title = str(it[0]).strip()
                desc = str(it[1]).strip()
                if title:
                    norm.append({title: desc})
            elif isinstance(it, str):
                s = it.strip()
                if s:
                    norm.append({s: ""})
        return norm

    result["projects"] = _to_single_kv_list(result["projects"])
    result["internships"] = _to_single_kv_list(result["internships"])
    result["achievements"] = _to_single_kv_list(result["achievements"])

    
    def _clean_list_str(items: list[Any]) -> list[str]:
        seen = set()
        out: list[str] = []
        for it in items:
            s = str(it).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    result["skills"] = _clean_list_str(result["skills"])
    result["certs"] = _clean_list_str(result["certs"])

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract resume information from a PDF using the OpenAI API and output JSON.",
    )
    parser.add_argument("pdf", help="Path to the resume PDF file")
    parser.add_argument(
        "-o",
        "--out",
        help="Output JSON file path (defaults to stdout if omitted)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name (e.g., gpt-4.1-mini, gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Reserved for future use (no effect)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models that support text generation and exit",
    )

    args = parser.parse_args(argv)

    try:
        if args.list_models:
            _ensure_openai_available()
            api_key = args.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set. Provide --api-key or set env var.")
            names = _list_openai_models(api_key)
            print(json.dumps(names, indent=2))
            return 0

        result = extract_resume(
            args.pdf,
            api_key=args.api_key,
            model_name=args.model,
            wait_timeout_sec=args.timeout,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
