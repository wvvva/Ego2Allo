from PIL import Image, ImageFont, ImageDraw
from typing import List, Optional, Union
import textwrap
import numpy as np
import torch
import trimesh

# Define decorator for each APC stage
def apc_stage(func):
    def wrapper(*args, **kwargs):
        print(f"[INFO] Running APC stage: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# ------------------------------------------------------------ #
# Conversation Utils
# Add message to conversation
def add_message(
    messages,
    role: str = "user",    # user, system
    text: str = None,
    image: Image.Image = None,
):
    '''
    Helper function to add a new message to the conversation
    '''
    if image is not None:
        new_message = {
            'role': role,
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': text}
            ]
        }
    else:
        new_message = {
            'role': role,
            'content': [
                {'type': 'text', 'text': text}
            ]
        }

    # Append the new message
    messages.append(new_message)

    return messages


# Make a image with text (for visualization)
def create_image_with_text(
    image,
    text,
    fontsize=15,
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
):
    '''
    Helper function to create a image with text
    '''
    # Load the input image
    image_width, image_height = image.size
    
    # Set up text area dimensions
    text_width = int(image_width)  # Adjust width for text
    total_width = image_width + text_width
    text_height = image_height

    # Create a new blank image with a white background
    image_with_text = Image.new('RGB', (total_width, text_height), 'white')
    image_with_text.paste(image, (0, 0))  # Paste the image on the left side
    
    # Add text to the right side
    draw = ImageDraw.Draw(image_with_text)

    try:
        font = ImageFont.truetype(font_path, fontsize)
    except IOError:
        font = ImageFont.load_default()

    # Automatically wrap the text to fit within the text area
    wrapped_text = textwrap.fill(
        text, 
        width=int(text_width / font.getlength('$'))
    )

    # Position the text on the image
    padding = 20
    text_x = image_width + padding
    text_y = padding  # Start padding from the top
    
    # Draw the text
    draw.text((text_x, text_y), wrapped_text, fill="black", font=font)
    
    return image_with_text

# ------------------------------------------------------------ #
# Rendering Utils
# ------------------------------------------------------------ #

def unit_cube(x1, y1, z1, x2, y2, z2, is_torch=True, device='cuda'):
    '''
    Create a unit cube
    '''
    verts = np.array([
        [x1, y1, z1],  # v0
        [x2, y1, z1],  # v1
        [x2, y2, z1],  # v2
        [x1, y2, z1],  # v3 
        [x1, y1, z2],  # v4
        [x2, y1, z2],  # v5
        [x2, y2, z2],  # v6
        [x1, y2, z2],  # v7
    ], dtype=np.float32)

    # 12 triangles (2 per face)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [0, 3, 7], [0, 7, 4],  # left
    ], dtype=np.int64)
    
    if is_torch:
        verts = torch.from_numpy(verts).to(device)
        faces = torch.from_numpy(faces).to(device)

    return verts, faces

def duplicate_verts(mesh: trimesh.Trimesh):
    '''
    Duplicate the vertices of a mesh
    '''
    verts = mesh.vertices[mesh.faces.reshape(-1), :]
    faces = np.arange(0, verts.shape[0])
    faces = faces.reshape(-1, 3)

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def perspective_grid(
    grid_size: int = 10,
    step: int = 10,
    max_depth: int = 1000,
):
    '''
    Visualize a tunnel-like grid to give a sense of depth
    '''
    lines = []

    # Fix x,y and vary z
    for x in range(-grid_size, grid_size + 1, step):
        for y in range(-grid_size, grid_size + 1, step):
            if x == 0 and y == 0:
                continue
            start = [x, y, -max_depth]
            end   = [x, y, grid_size]
            lines.append([start, end])

    # Fix x,z and vary y
    for x in range(-grid_size, grid_size + 1, step):
        for z in range(-max_depth, grid_size + 1, step):
            if x == 0:
                continue
            start = [x, -grid_size, z]
            end   = [x, grid_size, z]
            lines.append([start, end])

    # Fix y,z and vary x
    for y in range(-grid_size, grid_size + 1, step):
        for z in range(-max_depth, grid_size + 1, step):
            if y == 0:
                continue
            start = [-grid_size, y, z] 
            end   = [grid_size, y, z]
            lines.append([start, end])
            
    return lines

# ------------------------------------------------------------ #
# Conversation visualization (Generated with GPT-5)
# ------------------------------------------------------------ #
def visualize_conversation(
    items: List[dict],
    width: int = 1200,
    padding: int = 28,
    row_gap: int = 18,
    image_max_width: int = 220,
    font_path: Optional[str] = None,
    font_size: int = 22,
    text_bg: tuple = (246, 246, 246),
    canvas_bg: tuple = (255, 255, 255),
    text_color: tuple = (20, 20, 20),
    bubble_radius: int = 16,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Visualize a conversation as a single image.

    Parameters
    ----------
    items : list of dict
        Each dict should have:
            - "text": str            (required)
            - "image": str|PIL.Image (optional; local path or PIL Image, default None)
        Example:
            {"text": "Hello!", "image": None}
            {"text": "Here is the diagram", "image": "/path/to/img.png"}
    width : int
        Output image width in pixels.
    padding : int
        Outer padding around the canvas and inner padding for bubbles.
    row_gap : int
        Vertical gap between messages.
    image_max_width : int
        Maximum width allocated to the left-side image (if present).
    font_path : str | None
        Path to a TTF/OTF font. Falls back to a default if unavailable.
    font_size : int
        Base font size.
    text_bg : tuple
        RGB fill for the text bubble.
    canvas_bg : tuple
        RGB background for the canvas.
    text_color : tuple
        RGB color for the text.
    bubble_radius : int
        Corner radius for text bubbles.
    output_path : str | None
        If provided, saves the composed conversation image to this path.

    Returns
    -------
    PIL.Image.Image
        The rendered conversation image.
    """
    # --- helpers ---
    def load_font(size: int) -> ImageFont.FreeTypeFont:
        # Try a common font first; fall back to PIL's default
        try:
            if font_path and os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
            # DejaVuSans ships with many environments using Pillow
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
        # Returns (w, h) for the given text
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
        # Greedy word-wrapping that also splits very long words if needed
        words = text.replace("\n", " \n ").split(" ")
        lines = []
        line = ""
        for w in words:
            if w == "\n":
                lines.append(line)
                line = ""
                continue
            test = w if not line else f"{line} {w}"
            tw, _ = measure_text(draw, test, font)
            if tw <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                # If single word too long, split by characters
                ww, _ = measure_text(draw, w, font)
                if ww > max_width:
                    buf = ""
                    for ch in w:
                        tbuf = buf + ch
                        cw, _ = measure_text(draw, tbuf, font)
                        if cw <= max_width:
                            buf = tbuf
                        else:
                            if buf:
                                lines.append(buf)
                            buf = ch
                    line = buf
                else:
                    line = w
        if line:
            lines.append(line)
        return lines

    def rounded_rect(draw, xy, radius, fill):
        # Draw a rounded rectangle (Pillow >= 8.2 has rounded_rectangle natively)
        try:
            draw.rounded_rectangle(xy, radius=radius, fill=fill)
        except Exception:
            # Fallback: simple rectangle if rounded not available
            draw.rectangle(xy, fill=fill)

    def load_image(img: Union[str, Image.Image, None]) -> Optional[Image.Image]:
        if img is None:
            return None
        if isinstance(img, Image.Image):
            return img
        # Treat as local path
        return Image.open(img).convert("RGBA")

    # --- prepare drawing pass for measurements ---
    tmp = Image.new("RGB", (width, 200), canvas_bg)
    tmp_draw = ImageDraw.Draw(tmp)
    font = load_font(font_size)
    line_height = max(font.getbbox("Ag")[3] - font.getbbox("Ag")[1], font_size)  # robust line height
    line_gap = max(4, int(font_size * 0.2))

    # Pre-measure rows to calculate total height
    measured_rows = []
    total_height = padding  # top padding

    for idx, item in enumerate(items):
        text = str(item.get("text", ""))
        img = load_image(item.get("image"))
        
        # Calculate available space for text
        if img is not None:
            # Side-by-side layout: image on left, text fills remaining width
            left_block_w = image_max_width
            gutter = 16  # Space between image and text
            text_area_w = width - (padding * 2) - left_block_w - gutter
        else:
            # Full-width layout: text uses full width
            left_block_w = 0
            gutter = 0
            text_area_w = width - (padding * 2)

        # Wrap text within text area
        lines = wrap_text(tmp_draw, text, font, text_area_w)
        text_h = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
        text_h = max(text_h, line_height)  # at least one line height

        # Compute image placement height
        img_h = 0
        img_w = 0
        if img is not None:
            # Resize to fit the allocated width, keep aspect ratio
            iw, ih = img.size
            scale = min(image_max_width / iw, 1.0)  # never upscale width
            img_w = int(iw * scale)
            img_h = int(ih * scale)
            
            # Optional: Cap height to prevent extremely tall images
            # This keeps the layout more balanced
            max_img_h = min(400, text_h * 2)  # Reasonable max height
            if img_h > max_img_h:
                scale = max_img_h / img_h
                img_w = int(img_w * scale)
                img_h = int(img_h * scale)

        row_h = max(text_h, img_h) + padding  # inner padding for breathing room
        measured_rows.append({
            "lines": lines,
            "text_h": text_h,
            "img": img,
            "img_w": img_w,
            "img_h": img_h,
            "left_block_w": left_block_w,
            "text_area_w": text_area_w,
            "row_h": row_h,
            "gutter": gutter
        })
        total_height += row_h + row_gap

    total_height += padding - row_gap  # bottom padding (subtract last gap)

    # --- compose final image ---
    canvas = Image.new("RGB", (width, total_height), canvas_bg)
    draw = ImageDraw.Draw(canvas)

    y = padding
    for idx, row in enumerate(measured_rows):
        img = row["img"]
        img_w, img_h = row["img_w"], row["img_h"]
        row_h = row["row_h"]
        gutter = row["gutter"]

        x = padding
        # Draw image (left) if present
        if img is not None and img_w > 0 and img_h > 0:
            # Fit and paste with rounded corners
            img_resized = img.resize((img_w, img_h), Image.LANCZOS)
            # Rounded mask
            mask = Image.new("L", (img_w, img_h), 0)
            mdraw = ImageDraw.Draw(mask)
            rr = min(16, img_w // 8, img_h // 8)
            mdraw.rounded_rectangle([0, 0, img_w, img_h], radius=rr, fill=255)
            # Vertically center within row
            img_y = y + (row_h - img_h) // 2
            canvas.paste(img_resized, (x, img_y), mask)
            x += img_w + gutter  # move to start of text area

        # Draw text bubble - ensure it fills the remaining width
        bubble_w = row["text_area_w"]
        bubble_h = row["text_h"] + padding // 2  # inner bubble pad
        bubble_x0 = x
        bubble_y0 = y + (row_h - bubble_h) // 2
        bubble_x1 = bubble_x0 + bubble_w
        bubble_y1 = bubble_y0 + bubble_h
        
        # Ensure bubble doesn't exceed canvas bounds
        bubble_x1 = min(bubble_x1, width - padding)
        bubble_w = bubble_x1 - bubble_x0
        
        if idx < len(measured_rows) - 2:
            rounded_rect(draw, (bubble_x0, bubble_y0, bubble_x1, bubble_y1), bubble_radius, text_bg)
        else:
            rounded_rect(draw, (bubble_x0, bubble_y0, bubble_x1, bubble_y1), bubble_radius, (246, 246, 255))

        # Render wrapped text inside bubble
        tx = bubble_x0 + padding // 2
        ty = bubble_y0 + padding // 4
        for i, line in enumerate(row["lines"]):
            draw.text((tx, ty + i * (line_height + line_gap)), line, font=font, fill=text_color)

        y += row_h + row_gap

    if output_path:
        canvas.save(output_path)

    return canvas