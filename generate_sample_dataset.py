import os
import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_paper_texture(width, height):
    """Generates a realistic scanned exam notebook paper texture with subtle lighting vignette."""
    # Warm paper off-white base color (RGB: 250, 248, 242)
    base = np.full((height, width, 3), (250, 248, 242), dtype=np.uint8)
    
    # Add subtle paper grain noise
    grain = np.random.normal(0, 3.5, (height, width, 3)).astype(np.float32)
    paper = np.clip(base.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # Add lighting gradient vignette (slightly darker around edges like a real scanner/photo)
    X, Y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    dist = np.sqrt(X**2 + Y**2)
    vignette = 1.0 - 0.08 * dist
    vignette = np.stack([vignette]*3, axis=-1)
    
    scanned_paper = np.clip(paper.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
    return Image.fromarray(scanned_paper)

def draw_handwritten_line(draw, start, end, fill, width=2, jitter=1.5):
    """Draws a slightly shaky, natural line mimicking human hand strokes."""
    x1, y1 = start
    x2, y2 = end
    length = int(math.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    
    num_pts = max(3, length // 10)
    xs = np.linspace(x1, x2, num_pts)
    ys = np.linspace(y1, y2, num_pts)
    
    # Add minor perpendicular random jitter
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    perp_x, perp_y = -dy, dx
    
    points = []
    for i in range(num_pts):
        if i == 0 or i == num_pts - 1:
            j = 0
        else:
            j = np.random.uniform(-jitter, jitter)
        points.append((xs[i] + j * perp_x, ys[i] + j * perp_y))
        
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=fill, width=width)

def render_real_handwritten_page(page_num, content_blocks):
    width, height = 1200, 1650
    img = generate_paper_texture(width, height)
    draw = ImageDraw.Draw(img)

    # Native Windows Handwriting TTF Fonts
    try:
        font_inkfree_large = ImageFont.truetype(r"C:\Windows\Fonts\Inkfree.ttf", 26)
        font_inkfree_main = ImageFont.truetype(r"C:\Windows\Fonts\Inkfree.ttf", 22)
        font_segoe_script = ImageFont.truetype(r"C:\Windows\Fonts\segoesc.ttf", 21)
        font_segoe_print = ImageFont.truetype(r"C:\Windows\Fonts\segoepr.ttf", 22)
        font_segoe_bold = ImageFont.truetype(r"C:\Windows\Fonts\segoeprb.ttf", 22)
    except Exception as e:
        print("[Dataset Generator Warning] This generator needs Windows fonts Ink Free and Segoe Script.")
        font_inkfree_large = font_inkfree_main = font_segoe_script = font_segoe_print = font_segoe_bold = ImageFont.load_default()


    margin_x = 140
    header_h = 160

    # Draw Ruled Blue Notebook Lines
    blue_line_color = (210, 222, 238)
    for y in range(header_h + 35, height - 60, 42):
        draw_handwritten_line(draw, (0, y), (width, y), fill=blue_line_color, width=1, jitter=0.5)

    # Draw Red Margin Line
    draw_handwritten_line(draw, (margin_x, 0), (margin_x, height), fill=(225, 110, 110), width=3, jitter=0.8)
    # Double Line at Header
    draw_handwritten_line(draw, (0, header_h), (width, header_h), fill=(170, 170, 190), width=2, jitter=0.5)

    # Printed Exam Header Info
    draw.text((margin_x + 30, 25), "CNV AI LABS - TIMED TECHNICAL EVALUATION SCRIPT", fill=(30, 40, 70), font=font_segoe_bold)
    draw.text((margin_x + 30, 70), "Subject: Computer Networks & OS  |  Roll No: 2026-CS-8921", fill=(50, 60, 90), font=font_segoe_print)
    draw.text((margin_x + 30, 105), f"Student Answer Sheet — Page {page_num} of 2", fill=(70, 80, 110), font=font_segoe_print)

    current_y = header_h + 20

    # Real handwriting inks (Deep Fountain Pen Royal Blue & Red Pen)
    blue_ink = (18, 38, 110)
    black_ink = (30, 32, 45)
    red_marker_ink = (210, 40, 40)

    for block in content_blocks:
        b_type = block.get("type")

        if b_type == "question_header":
            # Margin Label (e.g. Q1., Q2.)
            q_label = block.get("label", "")
            draw.text((35, current_y), q_label, fill=(180, 25, 25), font=font_segoe_bold)
            
            # Question Title in main area
            q_title = block.get("title", "")
            draw.text((margin_x + 20, current_y), q_title, fill=(15, 25, 130), font=font_segoe_bold)
            current_y += 42

        elif b_type == "text":
            lines = block.get("lines", [])
            use_font = font_segoe_script if block.get("font") == "script" else font_inkfree_main
            for line in lines:
                x_offset = margin_x + 22 + np.random.randint(-2, 4)
                y_offset = current_y + np.random.randint(-1, 2)
                draw.text((x_offset, y_offset), line, fill=blue_ink, font=use_font)
                current_y += 42

        elif b_type == "crossed_out":
            line = block.get("line", "")
            x_offset = margin_x + 22
            draw.text((x_offset, current_y), line, fill=(100, 100, 120), font=font_inkfree_main)
            
            # Hand-drawn strikethrough lines with red pen
            bbox = draw.textbbox((x_offset, current_y), line, font=font_inkfree_main)
            w = bbox[2] - bbox[0]
            draw_handwritten_line(draw, (x_offset - 5, current_y + 14), (x_offset + w + 10, current_y + 16), fill=red_marker_ink, width=3, jitter=1.2)
            current_y += 42

        elif b_type == "margin_note":
            note = block.get("note", "")
            # Evaluator handwritten feedback box in left margin
            draw.rectangle([(10, current_y), (margin_x - 10, current_y + 65)], outline=(220, 90, 30), width=2)
            draw.text((15, current_y + 8), note, fill=(200, 60, 20), font=font_segoe_script)
            current_y += 75

        elif b_type == "diagram":
            diag_title = block.get("title", "DIAGRAM")
            layers = block.get("layers", [])

            box_x = margin_x + 80
            box_w = 440
            box_h = len(layers) * 34 + 50

            # Hand-drawn diagram outer box
            draw_handwritten_line(draw, (box_x, current_y), (box_x + box_w, current_y), fill=black_ink, width=2)
            draw_handwritten_line(draw, (box_x + box_w, current_y), (box_x + box_w, current_y + box_h), fill=black_ink, width=2)
            draw_handwritten_line(draw, (box_x + box_w, current_y + box_h), (box_x, current_y + box_h), fill=black_ink, width=2)
            draw_handwritten_line(draw, (box_x, current_y + box_h), (box_x, current_y), fill=black_ink, width=2)

            draw.text((box_x + 80, current_y + 8), f"[ {diag_title} ]", fill=(20, 30, 100), font=font_segoe_bold)

            layer_y = current_y + 42
            for l_text in layers:
                # Inner layer cell borders
                draw_handwritten_line(draw, (box_x + 15, layer_y), (box_x + box_w - 15, layer_y), fill=(80, 90, 140), width=1)
                draw.text((box_x + 30, layer_y + 4), l_text, fill=blue_ink, font=font_inkfree_main)
                layer_y += 34

            current_y += box_h + 30

    # Convert to OpenCV & apply camera rotation tilt (-0.4 deg)
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = cv_img.shape
    M = cv2.getRotationMatrix2D((w_img // 2, h_img // 2), -0.4, 1.0)
    cv_img = cv2.warpAffine(cv_img, M, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

    out_dir = r"c:\Users\Harsh raj\OneDrive\Desktop\CNV lab\sample_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"answer_sheet_page{page_num}.png")
    cv2.imwrite(out_path, cv_img)
    print(f"[Dataset Generator] Generated authentic handwritten answer sheet page {page_num} -> {out_path}")

def generate_all():
    page1_blocks = [
        {
            "type": "question_header",
            "label": "Q1.",
            "title": "Ans 1: TCP 3-Way Handshake Mechanism"
        },
        {
            "type": "text",
            "font": "inkfree",
            "lines": [
                "The TCP 3-way handshake is used to establish a reliable connection between client and server.",
                "1. SYN: Client sends a SYN segment with initial sequence number (ISN = X) to server.",
                "2. SYN-ACK: Server receives SYN and responds with SYN-ACK, acknowledging X+1 and sending seq Y.",
                "3. ACK: Client sends ACK packet acknowledging Y+1 to confirm connection.",
                "State transition goes from LISTEN -> SYN_SENT -> SYN_RCVD -> ESTABLISHED."
            ]
        },
        {
            "type": "question_header",
            "label": "Q2.",
            "title": "Ans 2: Process vs Thread Comparison"
        },
        {
            "type": "text",
            "font": "script",
            "lines": [
                "A Process is an independent program in execution with isolated virtual address space.",
                "A Thread is a lightweight unit of execution within a process that shares code and heap memory.",
                "Context switching in processes requires reloading page tables, so it is high overhead.",
                "Thread context switching is fast as they share address space.",
                "Fault Isolation: Process crash is isolated, but a faulty thread can crash the whole process."
            ]
        },
        {
            "type": "question_header",
            "label": "Q3.",
            "title": "Ans 3: OSI Reference Model 7 Layers & Functions"
        },
        {
            "type": "text",
            "font": "inkfree",
            "lines": [
                "The OSI model has 7 layers for network protocol standardization.",
                "Below is the layer architecture diagram:"
            ]
        },
        {
            "type": "diagram",
            "title": "OSI 7-LAYER ARCHITECTURE",
            "layers": [
                "7. Application Layer (HTTP, FTP)",
                "6. Presentation Layer (Encryption, Encoding)",
                "5. Session Layer (Dialog Control)",
                "4. Transport Layer (TCP/UDP, End-to-End)",
                "3. Network Layer (IP Routing)",
                "2. Data Link Layer (MAC Framing)",
                "1. Physical Layer (Bits, Cables)"
            ]
        },
        {
            "type": "text",
            "font": "script",
            "lines": [
                "[Continued on Page 2 for Layer detailed functions --->]"
            ]
        }
    ]

    page2_blocks = [
        {
            "type": "question_header",
            "label": "Q3.",
            "title": "Ans 3 (Continued from Page 1): Key Layer Functions"
        },
        {
            "type": "text",
            "font": "inkfree",
            "lines": [
                "Detailed functions of primary OSI layers:",
                "- Data Link Layer: Handles node-to-node framing, MAC addressing, and hardware error detection.",
                "- Network Layer: Performs logical IP addressing, packet forwarding, and path routing across subnetworks.",
                "- Transport Layer: Guarantees end-to-end reliability, sequence reordering, segmenting, and port mapping."
            ]
        },
        {
            "type": "question_header",
            "label": "Q4.",
            "title": "Ans 4: DNS Resolution Mechanism"
        },
        {
            "type": "margin_note",
            "note": "+1 Mark Bonus\nCheck RFC 1035"
        },
        {
            "type": "text",
            "font": "script",
            "lines": [
                "DNS (Domain Name System) translates human-readable hostnames into IP addresses.",
                "Types of resolution queries:"
            ]
        },
        {
            "type": "crossed_out",
            "line": "UDP is never used for DNS queries because IP packets get lost easily."
        },
        {
            "type": "text",
            "font": "inkfree",
            "lines": [
                "1. Recursive Resolution: Client delegates full lookup to local recursive resolver.",
                "   The resolver traverses Root -> TLD -> Authoritative DNS servers on behalf of client.",
                "2. Iterative Resolution: DNS server returns referral address of next server to client.",
                "   The client makes follow-up queries step-by-step until resolved."
            ]
        }
    ]

    render_real_handwritten_page(1, page1_blocks)
    render_real_handwritten_page(2, page2_blocks)

if __name__ == "__main__":
    generate_all()
