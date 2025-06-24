import streamlit as st
import torch
from PIL import Image
import io
import os  
from style_transfer import photorealistic_style_transfer, save_output
import time

st.set_page_config(page_title="Photorealistic Style Transfer", layout="wide")

st.title("🎨 Photorealistic Style Transfer")
st.write("""
This app demonstrates photorealistic style transfer using deep learning. Upload a content image 
and a style image to transfer the style while maintaining photorealism.
""")

# Create directories if they don't exist
for dir_name in ["uploaded_images", "output"]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_uploaded_image(image, path):
    """Save uploaded image after converting to RGB."""
    # Always convert to RGB mode
    image = image.convert('RGB')
    # Save with high quality
    image.save(path, 'JPEG', quality=95)

# Create two columns for content and style image uploads
col1, col2 = st.columns(2)

# Initialize variables
content_image = None
style_image = None
content_path = None
style_path = None

with col1:
    st.header("Content Image")
    content_file = st.file_uploader("Choose a content image...", type=["png", "jpg", "jpeg"])
    if content_file is not None:
        content_image = Image.open(content_file)
        st.image(content_image, caption="Content Image", use_container_width=True)
        # Save the uploaded content image
        content_path = os.path.join("uploaded_images", "content.jpg")
        save_uploaded_image(content_image, content_path)

with col2:
    st.header("Style Image")
    style_file = st.file_uploader("Choose a style image...", type=["png", "jpg", "jpeg"])
    if style_file is not None:
        style_image = Image.open(style_file)
        st.image(style_image, caption="Style Image", use_container_width=True)
        # Save the uploaded style image
        style_path = os.path.join("uploaded_images", "style.jpg")
        save_uploaded_image(style_image, style_path)

# Parameters sidebar
with st.sidebar:
    st.header("Transfer Parameters")
    quality_mode = st.select_slider(
        "Quality vs Speed",
        options=["Fastest", "Fast", "Balanced", "High Quality", "Ultra Quality"],
        value="Balanced",
        help="Adjusts multiple parameters to balance quality and speed"
    )
    
    # Adjust parameters based on quality mode
    if quality_mode == "Fastest":
        image_size = 256
        default_steps = 150
        max_steps = 200
        default_content_weight = 2.0
        default_style_weight = 8e4
        default_tv_weight = 2e-3
    elif quality_mode == "Fast":
        image_size = 320
        default_steps = 200
        max_steps = 300
        default_content_weight = 1.5
        default_style_weight = 9e4
        default_tv_weight = 1.5e-3
    elif quality_mode == "Balanced":
        image_size = 384
        default_steps = 300
        max_steps = 400
        default_content_weight = 1.0
        default_style_weight = 1e5
        default_tv_weight = 1e-3
    elif quality_mode == "High Quality":
        image_size = 448
        default_steps = 400
        max_steps = 500
        default_content_weight = 0.8
        default_style_weight = 1.2e5
        default_tv_weight = 8e-4
    else:  # Ultra Quality
        image_size = 512
        default_steps = 500
        max_steps = 600
        default_content_weight = 0.6
        default_style_weight = 1.5e5
        default_tv_weight = 5e-4
    
    # Advanced parameters section
    with st.expander("Advanced Parameters"):
        num_steps = st.slider("Number of Steps", 
                            min_value=100, 
                            max_value=max_steps, 
                            value=default_steps, 
                            step=50,
                            help="More steps generally give better results but take longer")
        
        content_weight = st.slider("Content Weight", 
                                min_value=0.1, 
                                max_value=5.0, 
                                value=default_content_weight, 
                                step=0.1,
                                help="Higher values preserve more content details")
        
        style_weight = st.slider("Style Weight", 
                               min_value=1e4, 
                               max_value=2e5, 
                               value=default_style_weight, 
                               step=1e4,
                               help="Higher values transfer more style characteristics")
        
        # Convert TV weight to more readable format (multiply by 1000 for display)
        tv_weight_display = st.slider("TV Weight (×10⁻³)", 
                                    min_value=0.1, 
                                    max_value=10.0, 
                                    value=default_tv_weight * 1000,
                                    step=0.1,
                                    help="Controls image smoothness (smaller values = smoother images)")
        # Convert back to actual TV weight
        tv_weight = tv_weight_display / 1000
        
        st.info(f"""
        Current Settings:
        - Image Size: {image_size}x{image_size}
        - Steps: {num_steps}
        - Content Weight: {content_weight:.1f}
        - Style Weight: {style_weight:.1e}
        - TV Weight: {tv_weight:.1e}
        """)

# Create output directory if it doesn't exist
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Process button
if st.button("Start Style Transfer", type="primary", disabled=not (content_file and style_file)):
    try:
        if content_image is None or style_image is None or content_path is None or style_path is None:
            st.error("Please upload both content and style images.")
            st.stop()
            
        st.header("Processing")
        
        # Create processing status container
        processing_status = st.empty()
        with processing_status:
            st.info("Preparing images...")
        
        # Time tracking
        start_time = time.time()
        
        # Process images based on quality mode
        content_image = content_image.convert('RGB').resize((image_size, image_size), Image.Resampling.LANCZOS)
        style_image = style_image.convert('RGB').resize((image_size, image_size), Image.Resampling.LANCZOS)
        
        # Save resized images
        content_image.save(content_path)
        style_image.save(style_path)
        
        
        # Run style transfer
        output = photorealistic_style_transfer(
            content_path, style_path,
            num_steps=num_steps,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight
        )

        with processing_status:
            st.info("Running style transfer...")
        
        # Save and display result
        output_path = os.path.join("output", "result.jpg")
        save_output(output, output_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clear processing status
        processing_status.empty()
        
        # Display result
        st.header("Result")
        result_image = Image.open(output_path)
        st.image(result_image, caption="Style Transfer Result", use_container_width=True)
        
        # Download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Result",
                data=file,
                file_name="style_transfer_result.jpg",
                mime="image/jpeg"
            )
        
        # Display processing information
        st.success(f"""
        Processing completed!
        - Quality mode: {quality_mode}
        - Image size: {result_image.size}
        - Processing time: {processing_time:.1f} seconds
        - Steps completed: {num_steps}
        - Content Weight: {content_weight}
        - Style Weight: {style_weight}
        - TV Weight: {tv_weight}
        """)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try again with different images or parameters.")

# Add information about the project
st.markdown("""
---
### How it works

This style transfer implementation uses:
1. **VGG19** network for feature extraction
2. **Content loss** to preserve structure
3. **Style loss** using Gram matrices
4. **Total variation loss** for smoothness

The algorithm maintains photorealism while transferring the style from one image to another.

### Tips for best results:
- Use high-quality images
- Try adjusting the weights if the results aren't as expected
- The process will automatically stop if the result converges
- Content and style images should have similar lighting conditions for best results
""") 
