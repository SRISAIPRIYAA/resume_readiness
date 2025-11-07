import streamlit as st
import base64

def get_base64_of_bin_file(bin_file):
    """Convert binary file to base64 string"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    """Set background image for Streamlit app"""
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        height: 100vh;
        overflow: hidden;
    }}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    # Page config
    st.set_page_config(
        page_title="LevelUp! AI Resume Review",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Set background image
    try:
        set_background('background.png')
    except FileNotFoundError:
        st.warning("Background image 'background.png' not found.")

    # --- UI Elements ---
    st.title("LevelUp! ✨")
    st.header("AI Resume Review")
    st.write("")

    # File uploader
    uploaded_file = st.file_uploader(
        "☁️ Drag & Drop or Click to Upload Your Resume PDF", 
        type=['pdf']
    )
    st.write("")

    # Domain selection
    domains = [
        "Select a domain...",
        "Web Development",
        "AI/ML",
        "Data Science",
        "Cybersecurity",
        "DevOps"
    ]
    selected_domain = st.selectbox(
        "Select the domain you're interested in:", 
        domains, 
        index=0
    )
    st.write("")

    # Analyze button
    if st.button("🚀 Analyze My Resume", use_container_width=True):
        # Input validation
        if uploaded_file is None:
            st.error("⚠️ Please upload your resume PDF first!")
        elif selected_domain == "Select a domain...":
            st.error("⚠️ Please select a domain!")
        else:
            # Save inputs to session_state
            st.session_state["selected_domain"] = selected_domain
            st.session_state["uploaded_resume_name"] = uploaded_file.name
            st.session_state["uploaded_resume_bytes"] = uploaded_file.getvalue()

            # Navigate to the results page - FIXED PATH
            st.switch_page("pages/result.py")

if __name__ == "__main__":
    main()