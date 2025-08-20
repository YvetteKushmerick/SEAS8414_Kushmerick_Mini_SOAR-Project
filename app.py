# app.py
import os
import time
import json
import streamlit as st
import pandas as pd
import re

from pycaret.classification import load_model as load_cls_model, predict_model as predict_cls_model
from pycaret.clustering import load_model as load_clu_model, predict_model as predict_clu_model
from genai_prescriptions import generate_prescription

# --- Page Configuration ---
st.set_page_config(
    page_title="GenAI-Powered Phishing SOAR",
    page_icon="🛡️",
    layout="wide"
)

CLF_PATH = 'models/phishing_url_detector'     # PyCaret adds .pkl
CLU_PATH = 'models/threat_actor_profiler'     # PyCaret adds .pkl
FEATURE_PLOT_PATH = 'models/feature_importance.png'
CLUSTER_MAP_PATH = 'models/threat_actor_profile_map.json'

# --- Load Model and Feature Plot ---
@st.cache_resource(show_spinner=False)
def load_assets():
    clf = load_cls_model(CLF_PATH) if os.path.exists(CLF_PATH + '.pkl') else None
    clu = load_clu_model(CLU_PATH) if os.path.exists(CLU_PATH + '.pkl') else None
    plot = FEATURE_PLOT_PATH if os.path.exists(FEATURE_PLOT_PATH) else None
    # Load cluster->actor mapping if present
    actor_map = {}
    if os.path.exists(CLUSTER_MAP_PATH):
        try:
            with open(CLUSTER_MAP_PATH, "r") as f:
                actor_map = {int(k): v for k, v in json.load(f).items()}
        except Exception:
            pass
    return clf, clu, plot, actor_map

model, cluster_model, feature_plot, ACTOR_MAP = load_assets()

if not model:
    st.error("Classifier not found. Train first (see container logs).")
    st.stop()

# --- Threat Actor Profile Mapping ---
CLUSTER_MAP = {
    0: {
        "name": "Organized Cybercrime",
        "desc": "Profit-driven groups focusing on large-scale phishing, ransomware, and financial fraud. They tend to run noisy, high-volume campaigns."
    },
    1: {
        "name": "State-Sponsored",
        "desc": "Advanced persistent actors pursuing espionage or disruption. They use sophisticated techniques, valid SSL certificates, and stealthy infrastructure."
    },
    2: {
        "name": "Hacktivist",
        "desc": "Ideologically motivated groups leveraging opportunistic attacks. They often deface sites or spread political messages using phishing lures."
    }
}

# --- Sidebar for Inputs ---
with st.sidebar:
    st.title("🔬 URL Feature Input")
    st.write("Describe the characteristics of a suspicious URL below.")

    # Using a dictionary to hold form values
    form_values = {
        'url_length': st.select_slider("URL Length", options=['Short', 'Normal', 'Long'], value='Long'),
        'ssl_state': st.select_slider("SSL Certificate Status", options=['Trusted', 'Suspicious', 'None'],
                                      value='Suspicious'),
        'sub_domain': st.select_slider("Sub-domain Complexity", options=['None', 'One', 'Many'], value='One'),
        'prefix_suffix': st.checkbox("URL has a Prefix/Suffix (e.g.,'-')", value=True),
        'has_ip': st.checkbox("URL uses an IP Address", value=False),
        'short_service': st.checkbox("Is it a shortened URL", value=False),
        'at_symbol': st.checkbox("URL contains '@' symbol", value=False),
        'abnormal_url': st.checkbox("Is it an abnormal URL", value=True),
        'has_political_keyword': st.checkbox("Contains political keyword(s)", value=True),
    }

    st.divider()
    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI", "Grok"])
    submitted = st.button("💥 Analyze & Initiate Response", use_container_width=True, type="primary")

# --- Main Page ---
st.title("🛡️ GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info("Please provide the URL features in the sidebar and click 'Analyze' to begin.")
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(feature_plot,
                 caption="Feature importance from the trained RandomForest model. This shows which features the model weighs most heavily when making a prediction.")

else:
    # --- Data Preparation and Risk Scoring ---
    input_dict = {
        'having_IP_Address': 1 if form_values['has_ip'] else -1,
        'URL_Length': -1 if form_values['url_length'] == 'Short' else (
            0 if form_values['url_length'] == 'Normal' else 1),
        'Shortining_Service': 1 if form_values['short_service'] else -1,
        'having_At_Symbol': 1 if form_values['at_symbol'] else -1,
        'double_slash_redirecting': -1,
        'Prefix_Suffix': 1 if form_values['prefix_suffix'] else -1,
        'having_Sub_Domain': -1 if form_values['sub_domain'] == 'None' else (
            0 if form_values['sub_domain'] == 'One' else 1),
        'SSLfinal_State': -1 if form_values['ssl_state'] == 'None' else (
            0 if form_values['ssl_state'] == 'Suspicious' else 1),
        'Abnormal_URL': 1 if form_values['abnormal_url'] else -1,
        'URL_of_Anchor': 0, 'Links_in_tags': 0, 'SFH': 0,
        'has_political_keyword': 1 if form_values['has_political_keyword'] else 0,
    }
    input_data = pd.DataFrame([input_dict])

    # Simple risk contribution for visualization
    risk_scores = {
        "Bad SSL": 25 if input_dict['SSLfinal_State'] < 1 else 0,
        "Abnormal URL": 20 if input_dict['Abnormal_URL'] == 1 else 0,
        "Prefix/Suffix": 15 if input_dict['Prefix_Suffix'] == 1 else 0,
        "Shortened URL": 15 if input_dict['Shortining_Service'] == 1 else 0,
        "Complex Sub-domain": 10 if input_dict['having_Sub_Domain'] == 1 else 0,
        "Long URL": 10 if input_dict['URL_Length'] == 1 else 0,
        "Uses IP Address": 5 if input_dict['having_IP_Address'] == 1 else 0,
        "Political Keyword Present": 5 if input_dict['has_political_keyword'] == 1 else 0,
    }
    risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Feature', 'Risk Contribution']).sort_values(
        'Risk Contribution', ascending=False)

    # --- Analysis Workflow ---
    with st.status("Executing SOAR playbook...", expanded=True) as status:
        st.write("▶️ **Step 1: Predictive Analysis** - Running features through classification model.")
        time.sleep(1)
        
        prediction = predict_cls_model(model, data=input_data)

        raw_label = prediction['prediction_label'].iloc[0]
        try:
            label_int = int(raw_label)
        except Exception:
            label_int = 1 if str(raw_label).lower() in ("1", "true", "malicious") else 0

        is_malicious = (label_int == 1)
        verdict = "MALICIOUS" if is_malicious else "BENIGN"
        
        
        st.write(f"▶️ **Step 2: Verdict Interpretation** - Model predicts **{verdict}**.")
        time.sleep(1)
        

        actor_profile = None
        if is_malicious and cluster_model is not None:
            st.write("▶️ **Step 2b: Threat Actor Attribution** - Running clustering model.")
            cluster_pred = predict_clu_model(cluster_model, data=input_data)

            # Find a usable column name across PyCaret versions
            label_col = next((c for c in ["Label", "Cluster", "Cluster Label", "label", "prediction_label"]
                            if c in cluster_pred.columns), None)
            if label_col is None:
                raise KeyError(f"No cluster label column found in prediction: {list(cluster_pred.columns)}")

            raw_lab = cluster_pred[label_col].iloc[0]

            # Normalize to an int, handling values like 0, "0", "Cluster 0"
            if isinstance(raw_lab, (int, float)):
                cluster_id = int(raw_lab)
            else:
                m = re.search(r"(\d+)", str(raw_lab))
                if not m:
                    raise ValueError(f"Could not parse cluster id from value: {raw_lab!r}")
                cluster_id = int(m.group(1))

            # Map cluster -> actor name/desc
            fallback = {0: "Organized Cybercrime", 1: "State-Linked Actor", 2: "Hacktivist"}
            actor_name = ACTOR_MAP.get(cluster_id, fallback.get(cluster_id, f"Cluster {cluster_id}"))
            DESCS = {
                "Organized Cybercrime": "Profit-driven, high-volume phishing, commodity kits and monetization.",
                "State-Linked Actor":   "Low-volume, targeted campaigns; stealthy infrastructure and valid SSL.",
                "Hacktivist":           "Ideology-driven, opportunistic lures; bursts around issues/events."
            }
            actor_desc = DESCS.get(actor_name, "Profile description not available.")

            st.write(f"Assigned to **{actor_name}** (Cluster {cluster_id}).")
            actor_profile = {"name": actor_name, "desc": actor_desc}
                
        elif is_malicious and cluster_model is None:
            st.warning("Threat attribution unavailable (clustering model missing). Train the clusterer to enable this.")


        if is_malicious:
            st.write(f"▶️ **Step 3: Prescriptive Analytics** - Engaging **{genai_provider}** for action plan.")
            try:
                prescription = generate_prescription(genai_provider, {k: v for k, v in input_dict.items()})
                status.update(label="✅ SOAR Playbook Executed Successfully!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Failed to generate prescription: {e}")
                prescription = None
                status.update(label="🚨 Error during GenAI prescription!", state="error")
        else:
            prescription = None
            status.update(label="✅ Analysis Complete. No threat found.", state="complete", expanded=False)



    # --- Tabs for Organized Output ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 **Analysis Summary**", 
        "📈 **Visual Insights**", 
        "📜 **Prescriptive Plan**", 
        "🕵️ **Threat Attribution**"
    ])

    with tab1:
        st.subheader("Verdict and Key Findings")
        if is_malicious:
            st.error("**Prediction: Malicious Phishing URL**", icon="🚨")
        else:
            st.success("**Prediction: Benign URL**", icon="✅")

        st.metric("Malicious Confidence Score",
                  f"{prediction['prediction_score'].iloc[0]:.2%}" if is_malicious else f"{1 - prediction['prediction_score'].iloc[0]:.2%}")
        st.caption("This score represents the model's confidence in its prediction.")

    with tab2:
        st.subheader("Visual Analysis")
        st.write("#### Risk Contribution by Feature")
        st.bar_chart(risk_df.set_index('Feature'))
        st.caption("A simplified view of which input features contributed most to a higher risk score.")

        if feature_plot:
            st.write("#### Model Feature Importance (Global)")
            st.image(feature_plot,
                     caption="This plot shows which features the model found most important *overall* during its training.")

    with tab3:
        st.subheader("Actionable Response Plan")
        if prescription:
            st.success("A prescriptive response plan has been generated by the AI.", icon="🤖")
            st.json(prescription, expanded=False)  # Show the raw JSON for transparency

            st.write("#### Recommended Actions (for Security Analyst)")
            for i, action in enumerate(prescription.get("recommended_actions", []), 1):
                st.markdown(f"**{i}.** {action}")

            st.write("#### Communication Draft (for End-User/Reporter)")
            st.text_area("Draft", prescription.get("communication_draft", ""), height=150)
        else:
            st.info("No prescriptive plan was generated because the URL was classified as benign.")

    with tab4:
        st.subheader("Threat Actor Attribution")
        if is_malicious and actor_profile:
            st.warning(f"Predicted Profile: **{actor_profile['name']}**", icon="🎯")
            st.write(actor_profile["desc"])
        else:
            st.info("Attribution is only available for URLs classified as malicious.")