import streamlit as st
import requests
import re
from openai import OpenAI
from datetime import datetime

# -------------------------------------------------------------------
# 1. INITIALIZATION & SECURE KEYS FROM STREAMLIT SECRETS
# -------------------------------------------------------------------
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    NASA_API_KEY = st.secrets["NASA_API_KEY"]
except KeyError:
    st.error("Missing Secrets Configuration! Please update `.streamlit/secrets.toml` with your token.")
    st.stop()

# Initialize GitHub Model Client
ai_client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_TOKEN,
)

# -------------------------------------------------------------------
# 2. NASA DATA FETCHING ENGINE (RAG TOOL)
# -------------------------------------------------------------------
def fetch_nasa_data(endpoint="apod"):
    """Fetches real-time astrophysics imagery and telemetry from various NASA endpoints."""
    base_urls = {
        "apod": f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}",
        "neo": f"https://api.nasa.gov/neo/rest/v1/feed?start_date={datetime.now().strftime('%Y-%m-%d')}&api_key={NASA_API_KEY}",
        "mars": f"https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/latest_photos?api_key={NASA_API_KEY}"
    }
    
    url = base_urls.get(endpoint)
    if not url:
        return None

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            data['source_type'] = endpoint
            return data
    except Exception as e:
        return {"error": str(e)}
    return None

# -------------------------------------------------------------------
# 3. TEXT PROCESSING UTILITIES
# -------------------------------------------------------------------
def process_ai_response(text, finalize=False):
    """Cleans <think> tags and formats LaTeX for Streamlit."""
    # Remove completed think tags
    processed = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # While streaming, remove the currently open think tag
    if not finalize:
        processed = re.sub(r"<think>.*", "", processed, flags=re.DOTALL)
    
    # Convert LaTeX brackets to Streamlit-friendly syntax
    processed = re.sub(r"\[\s*(.*?)\s*\]", r"$$\1$$", processed, flags=re.DOTALL)
    processed = re.sub(r"\(\s*([A-Za-z0-9_\\\^\{\}\(\)\s\,\.\-\+=\*\/]+?)\s*\)", r"$\1$", processed)
    
    return processed.strip()

# -------------------------------------------------------------------
# 3. STREAMLIT ULTIMATE SCIENTIFIC INTERFACE
# -------------------------------------------------------------------
st.set_page_config(page_title="AstroPhd-AI Discovery Lab", page_icon="🔭", layout="wide")

st.title("🔭 AstroPhD-AI Discovery Terminal")
st.caption("Space Analytics Engine • Powered by GitHub Models & NASA Data Vault")
st.markdown("---")

# Sidebar for controls and live NASA telemetry preview
with st.sidebar:
    st.header("🛰️ Data Stream Control")
    data_source = st.selectbox(
        "Select Active Telemetry Stream",
        ["Astronomy Picture of the Day (APOD)", "Near Earth Objects (Asteroids)", "Mars Rover (Curiosity)"],
        index=0
    )
    
    source_map = {
        "Astronomy Picture of the Day (APOD)": "apod",
        "Near Earth Objects (Asteroids)": "neo",
        "Mars Rover (Curiosity)": "mars"
    }

    if st.button("Initialize Data Uplink"):
        with st.spinner(f"Syncing with NASA {source_map[data_source].upper()} Node..."):
            data = fetch_nasa_data(source_map[data_source])
            if data and "error" not in data:
                st.session_state['nasa_context'] = data
                st.success(f"{source_map[data_source].upper()} Stream Active")
            else:
                st.error("Telemetry link failed. Check NASA API status.")

    st.divider()
    st.info("System Status: Operational\n\nDeep Space Network: Connected")

# -------------------------------------------------------------------
# 3. MAIN INTERFACE TABS
# -------------------------------------------------------------------
tab_chat, tab_deck = st.tabs(["🧬 Research Terminal", "🔭 Observation Deck"])

with tab_deck:
    st.header("Real-Time Observation Deck")
    if 'nasa_context' not in st.session_state:
        st.info("Awaiting telemetry initialization... Use the sidebar to query NASA systems.")
    else:
        ctx = st.session_state['nasa_context']
        source = ctx.get('source_type')
        
        if source == "apod":
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader(ctx.get('title', 'NASA APOD'))
                st.image(ctx.get('hdurl', ctx.get('url')), use_container_width=True)
            with col2:
                st.markdown("**Scientific Abstract:**")
                st.write(ctx.get('explanation'))
                st.caption(f"Date: {ctx.get('date')}")

        elif source == "neo":
            st.subheader("Near Earth Object (NEO) Tracking")
            neos = ctx.get('near_earth_objects', {})
            today = list(neos.keys())[0] if neos else None
            if today:
                st.write(f"Tracking **{len(neos[today])}** objects passing Earth today ({today})")
                for obj in neos[today][:5]: # Show top 5
                    with st.expander(f"Object: {obj['name']}"):
                        st.write(f"Estimated Diameter: {obj['estimated_diameter']['kilometers']['estimated_diameter_max']:.2f} km")
                        st.write(f"Hazardous: {'Yes' if obj['is_potentially_hazardous_asteroid'] else 'No'}")
                        st.write(f"Velocity: {obj['close_approach_data'][0]['relative_velocity']['kilometers_per_hour']} km/h")

        elif source == "mars":
            st.subheader("Mars Surface Reconnaissance")
            photos = ctx.get('latest_photos', [])
            if photos:
                cols = st.columns(3)
                for i, photo in enumerate(photos[:6]):
                    with cols[i % 3]:
                        st.image(photo['img_src'], caption=f"Rover: {photo['rover']['name']} | Cam: {photo['camera']['full_name']}")
                st.caption(f"Total photos retrieved: {len(photos)}")
            else:
                st.warning("No recent surface imagery available in this packet.")

# -------------------------------------------------------------------
# 4. CHAT SYSTEM INITIALIZATION
# -------------------------------------------------------------------
with tab_chat:
    SYSTEM_PROMPT = (
        "You are an expert senior astrophysicist. Answer users with high-level academic precision. "
        "You communicate using peer-reviewed vocabulary and formal logic. Express mathematical proofs "
        "using standard LaTeX formatting (using $$ for standalone block equations and $ for inline terms). "
        "Never explicitly mention or reveal that you are hiding your thinking process."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -------------------------------------------------------------------
    # 5. INTERACTIVE ANALYTICS ENGINE
    # -------------------------------------------------------------------
    if prompt := st.chat_input("Ask a PhD-level question or analyze active telemetry..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        enriched_prompt = prompt
        if 'nasa_context' in st.session_state:
            ctx = st.session_state['nasa_context']
            source = ctx.get('source_type')
            
            if source == "apod":
                context_str = f"Target: {ctx.get('title')}\nTelemetry: {ctx.get('explanation')}"
            elif source == "neo":
                context_str = f"Tracking {len(ctx.get('near_earth_objects', {}))} asteroids in close approach."
            elif source == "mars":
                context_str = "Analyzing latest high-resolution surface imagery from Curiosity Rover."
            else:
                context_str = "General NASA data stream active."
            
            enriched_prompt = (
                f"CONTEXT FROM NASA {source.upper()} STREAM:\n{context_str}\n\n"
                f"USER PHD INQUIRY: {prompt}"
            )
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            def run_inference(model_name, user_payload):
                full_response = ""
                stream = ai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload}
                    ],
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        display_text = process_ai_response(full_response)
                        if display_text.strip():
                            response_placeholder.markdown(display_text + "▌")
                return process_ai_response(full_response, finalize=True)

            try:
                # Attempt Primary Reasoning (DeepSeek-R1)
                final_content = run_inference("DeepSeek-R1", enriched_prompt)
            except Exception as primary_error:
                st.warning("🔄 Primary reasoning cluster congested. Scaling to secondary node...")
                try:
                    # Fallback to gpt-4o-mini
                    final_content = run_inference("gpt-4o-mini", enriched_prompt)
                except Exception as fallback_error:
                    st.error(f"Systems Offline: {str(fallback_error)}")
                    st.stop()

            # Finalize response
            response_placeholder.markdown(final_content)
            st.session_state.messages.append({"role": "assistant", "content": final_content})
