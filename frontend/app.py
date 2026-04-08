import streamlit as st
import requests
import json
import plotly.graph_objects as go

# Must be the first Streamlit command
st.set_page_config(
    page_title="Venture-Graph | Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Futuristic Dark Mode CSS Injection & Terminal Styling
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    /* Glowing Title */
    .title-glow {
        color: #58a6ff;
        text-shadow: 0 0 15px rgba(88, 166, 255, 0.6);
        font-family: 'Courier New', Courier, monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    /* Cyberpunk Terminal Container */
    .terminal-window {
        background-color: #010409;
        border: 1px solid #3fb950;
        border-radius: 5px;
        padding: 15px;
        height: 350px;
        overflow-y: auto;
        font-family: 'Consolas', 'Courier New', monospace;
        color: #3fb950;
        box-shadow: inset 0 0 10px rgba(63, 185, 80, 0.1);
    }
    .terminal-line {
        margin: 0;
        padding: 2px 0;
        font-size: 13px;
        border-bottom: 1px dashed #1f2b22;
    }
    .terminal-prefix { color: #8b949e; }
    .terminal-node { color: #a5d6ff; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding-bottom: 1rem;">
    <h1 style="margin-bottom: 0;">Venture-Graph</h1>
    <p style="font-size: 1.2rem; color: #64748b; margin-top: 0.5rem;">Automated Due Diligence & Market Feasibility Analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("`[SYSTEM OPTIMAL] :: MULTI-AGENT STATE GRAPH INITIALIZED`")
st.divider()
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 📥 Input Vector")
    user_idea = st.text_area(
        "DEFINE STARTUP HYPOTHESIS:",
        height=120,
        placeholder="e.g., An open-source headless CMS built in Rust for high-frequency trading platforms."
    )
    
    analyze_btn = st.button("EXECUTE DIRECTIVE", type="primary", use_container_width=True)
    
    st.markdown("<br>### 🧠 Core Processing Stream", unsafe_allow_html=True)
    stream_container = st.empty()

with col2:
    st.markdown("### 📊 Synthesized Output")
    report_container = st.empty()

def render_final_report(data: dict):
    """Renders the final JSON payload using Plotly and native Streamlit containers."""
    status = data.get("status", "UNKNOWN")
    
    # Color logic
    if status == "VALIDATE":
        hex_color = "#3fb950" # Green
    elif status == "NEEDS_WORK":
        hex_color = "#d29922" # Orange
    else:
        hex_color = "#f85149" # Red

    with report_container.container():
        st.markdown(f"## DECISION: <span style='color:{hex_color}; text-shadow: 0 0 15px {hex_color};'>{status}</span>", unsafe_allow_html=True)
        
        # Futuristic Gauge Chart for Confidence Score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = data.get("confidence_score", 0),
            title = {'text': "SYSTEM CONFIDENCE", 'font': {'color': '#8b949e', 'size': 14}},
            number = {'font': {'color': hex_color}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#8b949e"},
                'bar': {'color': hex_color},
                'bgcolor': "#010409",
                'borderwidth': 1,
                'bordercolor': "#30363d",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(248, 81, 73, 0.1)'},
                    {'range': [50, 80], 'color': 'rgba(210, 153, 34, 0.1)'},
                    {'range': [80, 100], 'color': 'rgba(63, 185, 80, 0.1)'}
                ]
            }
        ))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Courier New"})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"### 📝 Final Verdict\n> {data.get('final_verdict')}")
        
        # Fixed the DeltaGenerator bug by using native st.container(border=True)
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### 🌍 Market Assessment")
                m_data = data.get("market_assessment", {})
                if m_data.get("market_saturation_warning"):
                    st.warning("⚠️ High Saturation Detected")
                else:
                    st.success("✅ Blue Ocean Potential")
                st.caption(m_data.get('summary'))
                
        with c2:
            with st.container(border=True):
                st.markdown("#### ⚙️ Technical Feasibility")
                t_data = data.get("technical_feasibility", {})
                if t_data.get("is_buildable"):
                    st.success("✅ Architecture Viable")
                else:
                    st.error("❌ High Implementation Risk")
                st.metric("GitHub Repos Found", t_data.get("github_repos_found", 0))
                st.metric("Avg Competitor Stars", t_data.get("average_stars", 0))
                
        with st.container(border=True):
            st.markdown("#### 👾 Developer Sentiment (Hacker News)")
            st.info(data.get("developer_sentiment", "Awaiting telemetry..."))

# Execution Logic
if analyze_btn and user_idea:
    report_container.empty()
    
    with stream_container.container():
        # Initialize the terminal window UI
        status_html = "<div class='terminal-window'>"
        terminal_display = st.empty()
        terminal_display.markdown(status_html + "</div>", unsafe_allow_html=True)
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/evaluate",
                json={"idea": user_idea},
                stream=True
            )
            
            if response.status_code != 200:
                st.error(f"SYSTEM FAILURE: Backend API returned {response.status_code}")
            else:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            event_data = json.loads(decoded_line[6:])
                            
                            if event_data["type"] == "status":
                                msg = event_data["message"]
                                # Format as a terminal command line
                                new_line = f"<p class='terminal-line'><span class='terminal-prefix'>[root@venture-graph]~#</span> <span class='terminal-node'>{msg}</span></p>"
                                status_html += new_line
                                terminal_display.markdown(status_html + "</div>", unsafe_allow_html=True)
                                
                            elif event_data["type"] == "result":
                                new_line = f"<p class='terminal-line'><span class='terminal-prefix'>[root@venture-graph]~#</span> <span style='color: #3fb950;'>SYNTHESIS COMPLETE. PAYLOAD DEPLOYED.</span></p>"
                                status_html += new_line
                                terminal_display.markdown(status_html + "</div>", unsafe_allow_html=True)
                                
                                # Render the final UI
                                render_final_report(event_data["data"])
                                
        except requests.exceptions.ConnectionError:
            st.error("CONNECTION REFUSED. Ensure FastAPI backend is running on port 8000.")