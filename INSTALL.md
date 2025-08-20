# Mini_SOAR_Project Installation Guide
# This guide provides instructions on how to install and set up Mini_SOAR_Project on your system.

# The workflow uses PyCaret and Generative Artificial Intelligence in a Docker environment. The project structure is below. 
# mini-soar/
|── .github/workflows/ 	    # For GitHub Actions
|── README.md 		          # Project overview
|── INSTALL.md 		          # Installation guide
|── TESTING.md 		          # How to test 
|── Makefile 			          # Command shortcuts 
|── docker-compose.yml 	    # Docker orchestration
|── Dockerfile 		          # Container definition
|── requirements.txt 		    # Python libraries
|── train_model.py 		      # Predictive engine
|── genai_prescriptions.py 	# Prescriptive engine
|── app.py 			            # User Interface (UI) and main orchestrator

## 1. Prerequisites

# Before installing Mini_SOAR_Project, please ensure you have the following prerequisites installed:

# Docker Desktop at https://docs.docker.com/desktop/setup/install/windows-install/
# WSL CLI at https://learn.microsoft.com/en-us/windows/wsl/basic-commands
# Ubuntu partition at https://ubuntu.com/tutorials/install-ubuntu-desktop

#files within Project-Mini_SOAR folder in Ubuntu partition
#Docker
#Docker Compose
#Makefile
#The Dockerfile and docker-compose.yml are required to build the container image, including instructions for installing system and Python dependencies, and 
#defining services, networks, and #volumes, while the Makefile contains directives for the make build automation tool. The requirements.txt file 
#declares Python project dependencies, which ensures precise libraries in the development environment. 

#requirement.txt

#The requirement.txt This file is a standard way to declare a Python project's dependencies. The
pip install -r requirements.txt

#pandas, numpy: For data creation and manipulation.
#pycaret[full]: powerful, low-code automated machine learning library.
#streamlit: The framework for building our interactive web UI.
#google-generativeai, openai, requests: To communicate with the GenAI APIs for prescriptive analytics.

#command reads this file and installs the specifiedversions of each library.
#Git Clone
git clone https://github.com/YvetteKushmerick/SEAS8414_Kushmerick_Mini_SOAR-Project/tree/main

#Using WSL CLI, CMD
docker compose build
docker compose up

#ingest http://localhost:8501 into a browser

#edit train_model.py and app.py based on workflow requirements
