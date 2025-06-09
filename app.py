　　import streamlit as st
　　from PIL import Image
　　import numpy as np
　　import os
　　import tempfile
　　import cv2
　　from utils.image_utils import preprocess_drawing
　　from utils.model_utils import predict_profile, get_profile_data
　　from pdf2image import convert_from_path
　　from streamlit_drawable_canvas import st_canvas
　　
　　st.set_page_config(page_title="Technical Drawing Identifier", layout="wide")
　　st.title("✏️ AI-Based Technical Drawing Identifier")
　　
　　st.markdown("### Upload Drawing or Create One:")
　　
　　col1, col2 = st.columns(2)
　　
　　with col1:
　　    uploaded_file = st.file_uploader("Upload image or PDF", type=["jpg", "jpeg", "png", "pdf"])
　　
　　with col2:
　　    st.markdown("Draw a sketch (128x128 input)")
　　    canvas_result = st_canvas(
　　        fill_color="white",
　　        stroke_width=2,
　　        stroke_color="black",
　　        background_color="white",
　　        height=128,
　　        width=128,
　　        drawing_mode="freedraw",
　　        key="canvas",
　　    )
　　
　　input_image = None
　　
　　# If image/PDF uploaded
　　if uploaded_file:
　　    file_type = uploaded_file.type
　　    temp_dir = tempfile.mkdtemp()
　　    
　　    if file_type == "application/pdf":
　　        pdf_path = os.path.join(temp_dir, "input.pdf")
　　        with open(pdf_path, "wb") as f:
　　            f.write(uploaded_file.read())
　　        pages = convert_from_path(pdf_path, dpi=300)
　　        input_image = pages[0].convert("L")  # Take first page as grayscale
　　        st.image(input_image, caption="PDF Page 1 Preview", width=300)
　　        input_image.save("temp_input.jpg")
　　        img_array = preprocess_drawing("temp_input.jpg")
　　
　　    else:  # Image uploaded
　　        input_image = Image.open(uploaded_file).convert("L")
　　        st.image(input_image, caption="Uploaded Image", width=300)
　　        input_image.save("temp_input.jpg")
　　        img_array = preprocess_drawing("temp_input.jpg")
　　
　　# If sketch drawn
　　elif canvas_result.image_data is not None:
　　    drawn_img = canvas_result.image_data[:, :, 0]  # Extract grayscale
　　    img_array = cv2.resize(drawn_img, (128, 128)).astype(np.uint8)
　　    input_image = Image.fromarray(img_array)
　　    st.image(input_image, caption="Sketch Input", width=300)
　　
　　else:
　　    img_array = None
　　
　　# Process and predict
　　if img_array is not None:
　　    profile_name, confidence = predict_profile(img_array)
　　    st.subheader(f"🔍 Matched Profile: `{profile_name}` with {confidence*100:.2f}% confidence")
　　
　　    # PDF download
　　    pdf_path = f"data/{profile_name}.pdf"
　　    if os.path.exists(pdf_path):
　　        with open(pdf_path, "rb") as f:
　　            st.download_button(f"Download PDF Drawing for {profile_name}", f, file_name=f"{profile_name}.pdf")
　　
　　    # Profile data from Excel
　　    profile_info = get_profile_data(profile_name)
　　    if profile_info:
　　        st.markdown("### 📊 Profile Details:")
　　        for key, value in profile_info.items():
　　            st.write(f"**{key}**: {value}")
　　    else:
　　        st.warning("No matching data found in Excel for this profile.")
