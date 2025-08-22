# Mini-SOAR Project – Installation Guide

This guide provides instructions on how to install and set up **Mini-SOAR Project** on your system.

The workflow uses **PyCaret** and **Generative AI** in a **Docker environment**.  
Below is the project structure:

```
mini-soar/
│── .github/workflows/       # For GitHub Actions
│── README.md                # Project overview
│── INSTALL.md               # Installation guide
│── TESTING.md               # How to test
│── Makefile                 # Command shortcuts
│── docker-compose.yml       # Docker orchestration
│── Dockerfile               # Container definition
│── requirements.txt         # Python libraries
│── train_model.py           # Predictive engine
│── genai_prescriptions.py   # Prescriptive engine
│── app.py                   # User Interface (UI) and orchestrator
```

---

## 1. Prerequisites

Before installing **Mini-SOAR Project**, ensure the following are installed:

- [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)  
- [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)  
- [Ubuntu Partition](https://ubuntu.com/tutorials/install-ubuntu-desktop)  

Within the Ubuntu environment, you should have:

- **Docker**
- **Docker Compose**
- **Makefile**

The **Dockerfile** and **docker-compose.yml** are required to build the container image.  
- These handle installing dependencies, defining services, networks, and volumes.  
- The **Makefile** contains automation commands.  
- The **requirements.txt** file declares all Python dependencies, ensuring a reproducible development environment.  

---

## 2. Python Dependencies

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

### Libraries included:
- **pandas, numpy** → Data creation and manipulation  
- **pycaret[full]** → Automated machine learning  
- **streamlit** → Web application UI framework  
- **google-generativeai, openai, requests** → GenAI API communication  

---

## 3. Clone the Repository

```bash
git clone https://github.com/YvetteKushmerick/SEAS8414_Kushmerick_Mini_SOAR-Project.git
cd SEAS8414_Kushmerick_Mini_SOAR-Project
```

---

## 4. Build and Run with Docker

From inside the project directory, run:

```bash
docker compose build
docker compose up
```

---

## 5. Access the Application

Once the container is running, open your browser and navigate to:

```
http://localhost:8501
```

This will open the **Mini-SOAR Project UI** powered by Streamlit.

---

## 6. Makefile Reference

The **Makefile** provides shortcuts for common commands so you don’t have to type long Docker commands.  
Run these inside the project directory:

```bash
make build      # Builds the Docker image
make up         # Starts the Docker containers
make down       # Stops the containers
make restart    # Restarts the containers
make logs       # Shows container logs
make clean      # Removes stopped containers, images, and volumes
```

This simplifies your workflow and ensures consistent builds.

---

## 7. Post-Installation Notes

- Edit **train_model.py** and **app.py** as needed for your workflow requirements.  
- Use the **Makefile** to simplify build and run tasks.  
- Refer to [TESTING.md](./TESTING.md) for testing instructions.  

---

✅ Installation complete! You can now use Mini-SOAR Project to explore **automated incident detection, prediction, and response**.

