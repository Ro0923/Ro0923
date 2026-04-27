from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_report(confidences, violence, output="report.pdf"):
    c = canvas.Canvas(output, pagesize=A4)
    text = c.beginText(40, 800)

    text.textLine("AI Violence Detection Report")
    text.textLine("----------------------------------")
    text.textLine(f"Date: {datetime.now()}")
    text.textLine(f"Violence Detected: {violence}")
    text.textLine(f"Average Confidence: {sum(confidences)/len(confidences):.2f}")

    text.textLine("\nConfidence Timeline:")
    for i, conf in enumerate(confidences):
        text.textLine(f"Clip {i+1}: {conf:.2f}")

    c.drawText(text)
    c.save()
