import streamlit as st
import os
import io
import tempfile
from PIL import Image

# Import your pipeline (ensure your script is named `main.py` or adjust the import)
try:
    from main import build_graph
except ImportError:
    try:
        from food_agent_pipeline import build_graph
    except ImportError:
        build_graph = None

st.set_page_config(
    page_title="Fruit Classification and Color Estimation",
    page_icon="☘️",
    layout="wide"
)

st.title("☘️ Fruit Classification and Color Estimation")
st.markdown("Upload or capture an image to run Fruit Classification")

# Initialize session state for the active image
if "uploaded_image_bytes" not in st.session_state:
    st.session_state["uploaded_image_bytes"] = None

# --- Camera Dialog Popup ---
@st.dialog("Capture Image from Camera")
def camera_dialog():
    st.markdown("Position your item in front of the camera and click **Take Photo**.")
    camera_file = st.camera_input("Camera Feed")
    
    if camera_file is not None:
        if st.button("Use Captured Photo", type="primary"):
            st.session_state["uploaded_image_bytes"] = camera_file.getvalue()
            st.success("Photo captured successfully! You can close this dialog.")
            st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.header("Input Settings")
    input_mode = st.radio("Choose Input Method", ["📁 Upload Image", "📷 Capture Camera"])
    
    if input_mode == "📁 Upload Image":
        uploaded_file = st.file_uploader("Upload the image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state["uploaded_image_bytes"] = uploaded_file.getvalue()
    else:
        if st.button("Open Camera", use_container_width=True):
            camera_dialog()
            
    st.markdown("---")
    run_button = st.button("Run Analysis", type="primary", use_container_width=True, disabled=(st.session_state["uploaded_image_bytes"] is None))

# --- Main Layout ---
col_img, col_results = st.columns([1, 2], gap="large")

with col_img:
    st.subheader("Selected Image")
    if st.session_state["uploaded_image_bytes"] is not None:
        img = Image.open(io.BytesIO(st.session_state["uploaded_image_bytes"]))
        st.image(img, width="stretch")
    else:
        st.info("No image selected yet. Please upload or capture an image via the sidebar.")

with col_results:
    if run_button and st.session_state["uploaded_image_bytes"] is not None:
        if build_graph is None:
            st.error("Could not import `build_graph`. Make sure your pipeline script is in the same directory.")
        else:
            with st.spinner("Running Fruit Classification pipeline"):
                # Save session image bytes temporarily for the pipeline
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    img = Image.open(io.BytesIO(st.session_state["uploaded_image_bytes"]))
                    img.convert("RGB").save(tmp, format="JPEG", quality=90)
                    tmp_path = tmp.name
                
                try:
                    app = build_graph()
                    final = app.invoke({"image_path": tmp_path})
                    st.session_state["result"] = final.get("result", {})
                except Exception as e:
                    st.error(f"Error executing pipeline: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

# --- Render Results ---
if "result" in st.session_state:
    res = st.session_state["result"]
    st.markdown("---")
    
    if not res.get("is_fruit_or_vegetable", False):
        st.warning(res.get("message", "This is not a fruit or vegetable."))
        with st.expander("View Detection Details"):
            st.json(res.get("detection", {}))
    else:
        # Top Verdict Metrics
        st.subheader("Analysis Verdict")
        m1, m2, m3 = st.columns(3)
        m1.metric("Item Name", res.get("name", "").capitalize())
        m2.metric("Category", res.get("category", "").capitalize())
        m3.metric("Confidence", f"{res.get('confidence', 0) * 100:.0f}%")
        
        # Tabs for detailed breakdown
        # tab1, tab2, tab3 = st.tabs(["📊 Nutrition Profile", "💡 Health Benefits", "🤖 Agent Consensus"])
        tab1, tab2 = st.tabs(["📊 Nutrition Profile", "💡 Health Benefits"])
        
        with tab1:
            nutri = res.get("nutrition_per_100g", {})
            st.markdown(f"**Energy:** {nutri.get('calories_kcal_per_100g', 0)} kcal per 100g")
            
            # Macros
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Carbs", f"{nutri.get('carbs_g', 0)} g")
            c2.metric("Protein", f"{nutri.get('protein_g', 0)} g")
            c3.metric("Fat", f"{nutri.get('fat_g', 0)} g")
            c4.metric("Fiber", f"{nutri.get('fiber_g', 0)} g")
            c5.metric("Sugar", f"{nutri.get('sugar_g', 0)} g")
            
            v_col, m_col = st.columns(2)
            with v_col:
                st.markdown("#### Vitamins")
                for v in nutri.get("vitamins", []):
                    st.markdown(f"- **{v['name']}**: {v['amount_per_100g']}")
            with m_col:
                st.markdown("#### Minerals")
                for m in nutri.get("minerals", []):
                    st.markdown(f"- **{m['name']}**: {m['amount_per_100g']}")
                    
        with tab2:
            health = res.get("health_benefits", {})
            st.markdown("### May Support")
            for item in health.get("may_support", []):
                st.markdown(f"**{item.get('condition')}** (Evidence: `{item.get('evidence')}`)")
                st.markdown(f"> {item.get('how_it_may_help')}")
                
            st.markdown("### Cautions")
            for caution in health.get("cautions", []):
                st.markdown(f"- {caution}")
                
            st.info(res.get("disclaimer", ""))
            
        # with tab3:
        #     det = res.get("detection", {})
        #     st.markdown(f"**Resolution Method:** `{det.get('method')}`")
        #     st.markdown("#### Individual Agent Votes")
        #     for vote in det.get("agent_votes", []):
        #         with st.expander(f"Agent: `{vote.get('agent')}` — Category: {vote.get('category')} ({vote.get('confidence') * 100:.0f}%)"):
        #             st.markdown(f"**Detected Name:** {vote.get('name')}")
        #             st.markdown(f"**Reasoning:** {vote.get('reasoning')}")