
import base64
import io
from typing import List
from PIL import Image

def stitch_images(base64_images: List[str]) -> str:
    """
    Stitches a list of base64 encoded images vertically.
    
    Args:
        base64_images: List of base64 string images
        
    Returns:
        Base64 string of the stitched image
    """
    if not base64_images:
        return ""
    
    if len(base64_images) == 1:
        return base64_images[0]
        
    try:
        images = []
        total_height = 0
        max_width = 0
        
        # Decode and load all images
        for b64 in base64_images:
            # Clean header if present
            if "base64," in b64:
                b64 = b64.split("base64,")[1]
                
            img_data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
            total_height += img.height
            max_width = max(max_width, img.width)
            
        # Create new blank image
        stitched_img = Image.new('RGB', (max_width, total_height))
        
        # Paste images
        y_offset = 0
        for img in images:
            # Resize if width doesn't match (optional, but good for safety)
            if img.width != max_width:
                img = img.resize((max_width, int(img.height * (max_width / img.width))))
                
            stitched_img.paste(img, (0, y_offset))
            y_offset += img.height
            
        # Convert back to base64
        output_buffer = io.BytesIO()
        stitched_img.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        return base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"❌ Error stitching images: {e}")
        # Return first image as fallback
        return base64_images[0] if base64_images else ""
