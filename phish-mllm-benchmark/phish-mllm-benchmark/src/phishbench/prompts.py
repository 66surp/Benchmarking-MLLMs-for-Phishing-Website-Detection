
from __future__ import annotations
import os

ICR_SYSTEM = (
    "You are a cybersecurity analyst tasked with detecting phishing websites. "
    "Analyze the given webpage information carefully and output your answer in *valid JSON* following the schema."
)

ICR_USER_TEMPLATE = (
    "You are provided with one or more of the following inputs:\n"
    "1) URL string. 2) HTML-DOM text. 3) Screenshot image.\n"
    "Your task:\n"
    "- Check the URL for suspicious patterns or brand/domain mismatch.\n"
    "- Check the HTML for login/password forms or credential prompts.\n"
    "- Check the screenshot for brand/logo imitation or warning signs.\n"
    "- Combine the clues to decide. If evidence conflicts, reason briefly before deciding.\n\n"
    "Return STRICT JSON in this schema (no extra text):\n"
    "{\n"
    "  \"label\": \"phishing or legit\",\n"
    "  \"confidence\": 0.0-1.0,\n"
    "  \"evidence\": {\n"
    "    \"url_spans\": [\"suspicious substrings\"],\n"
    "    \"dom_selectors\": [\"css/xpath/id or phrases\"],\n"
    "    \"image_boxes\": [[x1,y1,x2,y2]]\n"
    "  },\n"
    "  \"rationale\": \"one or two sentences\"\n"
    "}\n\n"
    "Inputs provided now:\n{inputs_block}\n"
)

def build_inputs_block(url: str | None, html: str | None, image_path: str | None, html_max_chars: int = 3500) -> str:
    parts = []
    if url:
        parts.append(f"URL: {url}")
    if html:
        truncated = html if len(html) <= html_max_chars else (html[:html_max_chars] + "... [TRUNCATED]")
        parts.append(f"HTML-DOM (truncated):\n{truncated}")
    if image_path:
        parts.append(f"IMAGE: <attached screenshot: {os.path.basename(image_path)}>\n")
    return "\n\n".join(parts) if parts else "(no inputs)"
