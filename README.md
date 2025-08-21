# 🛡️ Mini-SOAR Project  

A research-oriented proof-of-concept **Security Orchestration, Automation, and Response (SOAR)** platform.  
This project demonstrates how **machine learning**, **unsupervised clustering**, and **generative AI** can be integrated into a single workflow to enhance phishing incident detection, attribution, and response.  

---

## 📖 Overview  
The **Mini-SOAR Project** simulates how a modern SOC (Security Operations Center) can use AI to:  
- Detect phishing attempts using supervised classification.  
- Attribute malicious URLs to **threat actor profiles** using clustering.  
- Generate **step-by-step incident response playbooks** using large language models (LLMs).  

The system is implemented as an interactive **Streamlit dashboard**, designed for **academic exploration and experimentation**.  

---

## 🎯 Objectives  
- Apply **classification models** to distinguish malicious vs. benign URLs.  
- Use **clustering** to attribute malicious URLs to representative **threat actors**.  
- Evaluate the utility of **LLMs** in producing concise, prescriptive response playbooks.  
- Provide an **educational tool** for cybersecurity and AI coursework.  

---

## ⚙️ Methodology  

1. **Data & Features**  
   - Synthetic phishing/benign datasets generated with URL-based features (SSL state, length, domain, etc.).  

2. **Classification (Supervised)**  
   - PyCaret classification pipeline for **Malicious vs. Benign** prediction.  

3. **Clustering (Unsupervised)**  
   - PyCaret clustering for attribution into profiles (e.g., **State-Sponsored**, **Organized Cybercrime**, **Hacktivists**).  

4. **Playbook Generation**  
   - Generative AI (Gemini / OpenAI API) produces 3–4 step playbooks for Tier 1 SOC analysts.  

5. **Dashboard (Streamlit)**  
   - User interface for data submission, model predictions, and AI-driven recommendations.  

---

## 📂 Project Structure  
```
# mini-soar/
|── .github/workflows/ 	   # For GitHub Actions
|── README.md 		         # Project overview
|── INSTALL.md 		      # Installation guide
|── TESTING.md 		      # How to test 
|── Makefile 			      # Command shortcuts 
|── docker-compose.yml 	   # Docker orchestration
|── Dockerfile 		      # Container definition
|── requirements.txt 	   # Python libraries
|── train_model.py 		   # Predictive engine
|── genai_prescriptions.py # Prescriptive engine
|── app.py 			   # User Interface (UI) and main orchestrator
```

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.9+  
- PyCaret  
- Streamlit  
- API key for Gemini or OpenAI (for playbook generation)  

### Installation  
```bash
# Clone repository
git clone https://github.com/yourusername/mini-soar.git
cd mini-soar

# Install dependencies
pip install -r requirements.txt
```

### Run Application  
```bash
streamlit run app.py
```

---

## 📊 Workflow Demonstration  
1. Analyst submits suspicious URL features.  
2. System classifies input as **Malicious** or **Benign**.  
3. If malicious → Clustering assigns it to a **threat actor profile**.  
4. Generative AI produces a **prescriptive response playbook**.  

---

## 🔬 Academic Contribution  
This project highlights the **synergy between AI and cybersecurity operations**, specifically:  
- The feasibility of **AI-driven SOAR workflows**.  
- The role of **clustering for attribution** in phishing detection.  
- The potential of **LLM-based prescriptive guidance** for Tier 1 analysts.  

---

## 📌 Future Work  
- Validate against **real-world phishing datasets**.  
- Expand **threat actor taxonomy**.  
- Explore integration with SIEM/SOAR platforms (e.g., Splunk, Cortex XSOAR).  
- Conduct user studies on **analyst trust and usability**.  

---
## 📜 License  
Released under the **MIT License**. For research and educational use only.  

---
⚠️ **Disclaimer:** This is a research prototype. It is not intended for production SOC deployment.  


 
