import ollama
from datetime import datetime

def generate_report(defects: list, image_filename: str = "steel_sample"):
    if not defects:
        return _no_defect_report(image_filename)

    # Build defect summary for the prompt
    # Example: - Crazing: 87.5% confidence, Severity: High
    defect_lines = "\n".join([
        f"- {d['class'].replace('_', ' ').title()}: "
        f"{d['confidence']}% confidence, Severity: {d['severity']}"
        for d in defects
    ])

    # Counts how many defects fall into each severity level
    severity_counts = {}
    for d in defects:
        severity_counts[d['severity']] = severity_counts.get(d['severity'], 0) + 1

    prompt = f"""You are an expert industrial quality assurance engineer specializing in steel surface inspection.

A YOLOv8 computer vision model has analyzed a steel surface sample and detected the following defects:

Inspection  : {datetime.now().strftime("%Y-%m-%d %H:%M")} 
Total defects detected: {len(defects)}

Detected Defects:
{defect_lines}

Severity Summary:
{chr(10).join([f"- {k}: {v} defect(s)" for k, v in severity_counts.items()])}

Please generate a professional industrial inspection report that includes:
1. Inspection Date
2. Detected Defects with confidence percentage in brackets (ordered). 
3. Executive summary (1-2 lines) that explains number of surface defects and what the defects might affect
4. Severity 
5. Recommended Actions (specific steps the manufacturing team should take)
Do not add any place to sign or any instructions to acknowledge the report.
Use professional industrial language. Be specific and actionable."""

    response = ollama.chat(
        model    = "llama3.2",
        messages = [{"role": "user", "content": prompt}] # Ollama processes messages and returns a response object
    )

    return response["message"]["content"]

# underscore prefix means it's internal and not meant to be called from outside this file
def _no_defect_report(image_filename: str) -> str:
    return f"""
## Inspection Report — {image_filename}
**Inspection Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

### Executive Summary
No surface defects were detected in this steel sample. The surface meets quality standards based on automated visual inspection.
"""