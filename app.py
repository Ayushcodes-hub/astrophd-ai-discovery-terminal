import streamlit as st
import requests
import re
from openai import OpenAI

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
def fetch_nasa_data():
    """Fetches real-time astrophysics imagery and scientific telemetry from NASA."""
    url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        return {"error": str(e)}
    return None

# -------------------------------------------------------------------
# 3. STREAMLIT ULTIMATE SCIENTIFIC INTERFACE
# -------------------------------------------------------------------
st.set_page_config(page_title="AstroPhd-AI Discovery Lab", page_icon="🔭", layout="wide")

st.title("🔭 AstroPhD-AI Discovery Terminal")
st.caption("Space Analytics Engine • Powered by GitHub Models & NASA Data Vault")
st.markdown("---")

# Sidebar for controls and live NASA telemetry preview
with st.sidebar:
    st.header("🛰️ NASA Live Data Node")
    if st.button("Query NASA Live System"):
        with st.spinner("Connecting to NASA Deep Space Network..."):
            data = fetch_nasa_data()
            if data and "error" not in data:
                st.session_state['nasa_context'] = data
                st.success("Data Synthesized Successfully!")
            else:
                st.error("Telemetry failed. Using cached static context.")
                
    if 'nasa_context' in st.session_state:
        ctx = st.session_state['nasa_context']
        st.subheader(f"Current Target: {ctx.get('title', 'Unknown')}")
        if 'hdurl' in ctx:
            st.image(ctx['hdurl'], use_container_width=True)
        st.info(f"Target Date: {ctx.get('date', 'N/A')}")

# Initialize chat logs
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system", 
            "content": "You are a senior astrophysicist. You communicate using peer-reviewed vocabulary, formal logic, and express mathematical proofs using standard LaTeX formatting (using $$ for block equations and $ for inline terms) when explaining astronomical phenomena."
        }
    ]

# Render chat history cleanly (skipping the system prompt)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# -------------------------------------------------------------------
# 4. CHAT INTERACTIVE LOGIC (ROBUST STREAM HANDLING)
# -------------------------------------------------------------------
if prompt := st.chat_input("Ask a PhD-level question or analyze active telemetry..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    enriched_prompt = prompt
    if 'nasa_context' in st.session_state:
        ctx = st.session_state['nasa_context']
        enriched_prompt = (
            f"CONTEXT FROM NASA TELEMETRY DEEP NET:\n"
            f"Target Object Title: {ctx.get('title')}\n"
            f"Scientific Abstract/Telemetry: {ctx.get('explanation')}\n\n"
            f"USER PHD INQUIRY: {prompt}"
        )
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Main Stream Loop Configuration
            stream = ai_client.chat.completions.create(
                model="DeepSeek-R1", 
                messages=[
                    {"role": "system", "content": st.session_state.messages[0]["content"]},
                    {"role": "user", "content": enriched_prompt}
                ],
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        full_response += delta.content
                        
                        # 1. Clean raw reasoning block markers
                        display_text = full_response.replace("<think>", "**Thinking Process:**\n")
                        display_text = display_text.replace("</think>", "\n\n**Analysis:**\n")
                        
                        # 2. Dynamic regex cleanup to convert text-brackets to formal Streamlit LaTeX
                        # Converts [ equation ] to $$ equation $$
                        display_text = re.sub(r'\[\s*(.*?)\s*\]', r'$$\1$$', display_text)
                        # Converts ( variable ) to $ variable $
                        display_text = re.sub(r'\(\s*([A-Za-z0-9_\\\^\{\}\(\)\s\,\.\-\+=\*\/]+?)\s*\)', r'$\1$', display_text)
                        
                        response_placeholder.markdown(display_text + "▌")
            
            # Final clean render processing
            final_text = full_response.replace("<think>", "**Thinking Process:**\n")
            final_text = final_text.replace("</think>", "\n\n**Analysis:**\n")
            final_text = re.sub(r'\[\s*(.*?)\s*\]', r'$$\1$$', final_text)
            final_text = re.sub(r'\(\s*([A-Za-z0-9_\\\^\{\}\(\)\s\,\.\-\+=\*\/]+?)\s*\)', r'$\1$', final_text)
            
            response_placeholder.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
            
        except Exception as primary_error:
            st.warning("🔄 Primary reasoning cluster congested. Scaling to secondary node...")
            full_response = ""
            try:
                # Robust failover fallback to gpt-4o-mini
                fallback_stream = ai_client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[
                        {"role": "system", "content": st.session_state.messages[0]["content"]},
                        {"role": "user", "content": enriched_prompt}
                    ],
                    stream=True
                )
                for chunk in fallback_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_response += delta.content
                            response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as fallback_error:
                st.error(f"Execution Halt: Stream Interrupted.\nPrimary Node: {str(primary_error)}\nFallback Node: {str(fallback_error)}")