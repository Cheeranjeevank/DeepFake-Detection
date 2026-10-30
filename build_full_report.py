import os
import base64
import subprocess
from pypdf import PdfReader

def get_b64(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            ext = os.path.splitext(path)[1].replace('.', '').lower()
            if ext == 'jpg': ext = 'jpeg'
            return f"data:image/{ext};base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return ""

# Load all 8 image assets
img_arch = get_b64('/Users/cheeranjeevan.k/Documents/DL/deepguard/outputs/report_images/system_architecture.png')
img_wf = get_b64('/Users/cheeranjeevan.k/Documents/DL/deepguard/outputs/report_images/functional_workflow.png')
img_home = get_b64('/Users/cheeranjeevan.k/Documents/DL/deepguard/outputs/report_images/home_page_1790779313749.png')
img_upload = get_b64('/Users/cheeranjeevan.k/Documents/DL/deepguard/outputs/report_images/image_upload_page_1790779528012.png')
img_cam = get_b64('/Users/cheeranjeevan.k/Documents/DL/deepguard/outputs/report_images/gradcam_demo_1790781358873.jpg')
img_bottom = get_b64('/Users/cheeranjeevan.k/Documents/DL/deepguard/outputs/report_images/home_page_bottom_1790779342802.png')
img_hist = get_b64('/Users/cheeranjeevan.k/Documents/DL/deepguard/outputs/graphs/training_history.png')
img_roc = get_b64('/Users/cheeranjeevan.k/Documents/DL/deepguard/outputs/report_images/roc_confusion_matrix.png')

print("Loaded all images.")

# Common styles
css = """
@page {
    size: A4;
    margin: 0;
}
* {
    box-sizing: border-box;
}
body {
    margin: 0;
    padding: 0;
    font-family: 'Times New Roman', Times, Georgia, serif;
    font-size: 12pt;
    line-height: 1.5;
    color: #111;
    background: #fff;
}
.page {
    width: 210mm;
    height: 297mm;
    max-height: 297mm;
    padding: 24mm 20mm 20mm 25mm; /* Anna Univ standard margin: 25mm left for binding */
    page-break-after: always;
    break-after: page;
    position: relative;
    overflow: hidden;
}
.page-num {
    position: absolute;
    top: 14mm;
    right: 20mm;
    font-size: 11pt;
    font-family: 'Times New Roman', Times, serif;
}
.cover-page {
    width: 210mm;
    height: 297mm;
    max-height: 297mm;
    padding: 20mm 18mm;
    page-break-after: always;
    break-after: page;
    position: relative;
    overflow: hidden;
}
.cover-inner {
    border: 3px double #000;
    width: 100%;
    height: 100%;
    padding: 16mm 14mm;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
h1, h2, h3, h4 {
    font-family: 'Times New Roman', Times, serif;
    color: #000;
    font-weight: bold;
}
.chapter-num {
    font-size: 14pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    margin-bottom: 4pt;
}
.chapter-title {
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    margin-bottom: 18pt;
}
h2 {
    font-size: 13pt;
    margin-top: 14pt;
    margin-bottom: 6pt;
    text-transform: uppercase;
}
h3 {
    font-size: 12pt;
    margin-top: 10pt;
    margin-bottom: 4pt;
}
p {
    text-align: justify;
    text-justify: inter-word;
    margin-top: 0;
    margin-bottom: 8pt;
    text-indent: 1.5em;
}
.no-indent {
    text-indent: 0 !important;
}
ul, ol {
    margin-top: 2pt;
    margin-bottom: 8pt;
    padding-left: 1.8em;
    text-align: justify;
}
li {
    margin-bottom: 4pt;
    line-height: 1.4;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0;
    font-size: 10pt;
}
th, td {
    border: 1px solid #222;
    padding: 5pt 6pt;
    text-align: left;
    vertical-align: middle;
}
th {
    background-color: #f1f5f9;
    font-weight: bold;
    text-align: center;
}
.figure-container {
    text-align: center;
    margin: 10pt 0;
}
.figure-container img {
    max-width: 95%;
    max-height: 140mm;
    height: auto;
    border: 1px solid #94a3b8;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12);
}
.figure-caption {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 8pt;
    text-align: center;
    color: #0f172a;
}
.figure-desc {
    font-size: 9.5pt;
    font-style: italic;
    color: #475569;
    margin-top: 3pt;
    text-align: center;
}
pre.code-block {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3.5px solid #2563eb;
    padding: 6pt 8pt;
    font-family: 'Courier New', Courier, monospace;
    font-size: 8pt;
    line-height: 1.28;
    overflow-x: auto;
    margin: 6pt 0 8pt 0;
    border-radius: 3px;
    white-space: pre;
}
.math-eq {
    text-align: center;
    margin: 6pt 0;
    font-style: italic;
    font-size: 10.5pt;
}
.signatures-table {
    width: 100%;
    border: none;
    margin-top: 25pt;
}
.signatures-table td {
    border: none;
    padding: 6pt;
    vertical-align: top;
}
"""

print("Writing pages...")
