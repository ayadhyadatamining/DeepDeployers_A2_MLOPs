# DeepDeployers_A2_MLOPs

# Sentiment Analysis API with Fine-Tuned BART Model

## 📌 Project Overview

This project is a sentiment analysis system built using FastAPI, a fine-tuned BART model, and a frontend with Jinja2 templates. It allows users to analyze the sentiment of text inputs and provides an API for training and predictions. The model is fine-tuned for binary sentiment classification (positive/negative).

---

## 🎥 Project Demonstration

Watch the project demonstration video here:  
[![Project Demonstration](https://img.shields.io/badge/Project%20Demo-Click%20Here-blue?style=for-the-badge)](https://drive.google.com/file/d/13A20RC270TYE0mmHpBlIXURGJ_JmIYtH/view?usp=sharing)


## 📂 Project Structure

```bash
├── .dockerignore             # Docker ignore file
├── .gitignore                # Git ignore file
├── Best_model_param          # Stores the best model parameters
├── docker-compose.yml        # Docker Compose setup
├── Final_trained_model       # Directory for storing trained models
├── mlruns                    # MLflow tracking runs
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies list
├── Research                  # Research notes and experiments
├── server.py                 # FastAPI application server
├── src                       # Source code directory
├── static                    # Static files (CSS, JS, images, etc.)
├── templates                 # HTML template files for frontend
├── venv                      # Python virtual environment
├── __pycache__               # Compiled Python files
```

---

## 🚀 API Endpoints

| HTTP Method | Endpoint             | Description                                                                                                                      |
| ----------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `GET`       | `/`                  | **Landing Page** - Loads the frontend project landing page with a start button.                                                  |
| `GET`       | `/sentiment`         | **Sentiment Analysis Page** - Loads the sentiment analysis page where users can input text.                                      |
| `POST`      | `/analyze_sentiment` | **Analyze Sentiment** - Receives a text input, processes it through the fine-tuned BART model, and returns the sentiment result. |
| `GET`       | `/model_parameters`  | **Model Parameters** - Returns the best training parameter details.                                                              |
| `POST`      | `/model_training`    | **Train Model** - Accepts a dataset file path and starts the fine-tuning of the BART model.                                      |

---

## 🔧 Running the Project

### 📌 Local Setup

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Run the FastAPI server locally**:

```bash
uvicorn server:app --reload
```

3. Open the API docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🐳 Docker Setup

### 📌 Build & Run the Docker Container

1. **Build the Docker image**:

```bash
docker build -t sentiment-analysis .
```

2. **Run the Docker container**:

```bash
docker run -p 8000:8000 sentiment-analysis
```

3. **Access the application**:

- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Web UI: [http://localhost:8000](http://localhost:8000)

---

## 📊 Model Training & MLflow Tracking

The model fine-tuning process is logged and tracked using **MLflow**. To track the training:

1. **Run MLflow tracking server**:

```bash
mlflow ui
```

2. Open MLflow UI at: [http://localhost:5000](http://localhost:5000)

---

## 🔄 Router Details

| Route                | Functionality                                     |
| -------------------- | ------------------------------------------------- |
| `/`                  | Loads the landing page with a start button.       |
| `/sentiment`         | Loads the sentiment analysis page.                |
| `/analyze_sentiment` | Accepts text input and returns sentiment results. |
| `/model_parameters`  | Fetches the best training parameters.             |
| `/model_training`    | Starts the model fine-tuning process.             |

---

## 📌 Technologies Used

- **FastAPI** for API development 🚀
- **Transformers (Hugging Face)** for model fine-tuning 🤗
- **Torch (PyTorch)** for deep learning computations 🔥
- **MLflow** for tracking model training 📊
- **Docker** for containerization 🐳
- **Uvicorn** for running the FastAPI app ⚡

---

## 🛠 Future Enhancements

- Add support for multi-class sentiment analysis.
- Implement a real-time monitoring dashboard for API requests.
- Extend the model to support multiple languages.

---

### 💡 Contributors

| Member                     | Role & Responsibilities                                                                                                                                              |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Arup Das**               | Data & Model Development  - Dataset preprocessing and cleaning  - Implement TF-IDF + classifier pipeline  - BERT model fine-tuning  - Model evaluation metrics       |
| **Abhishek Prasad Nonia**  | API & Infrastructure  - Design REST API endpoints  - Implement request/response handling  - Database integration for model parameters  - Authentication system setup |
| **Piyush Sudhir Shobhane** | Deployment & Testing  - Docker containerization  - CI/CD pipeline setup  - Unit/Integration testing  - Performance benchmarking                                      |

---

### 📜 License

This project is licensed under the MIT License.



### Submission Deadline - 26th Feb 2025	
	
### ML model to develop	Dataset
"1. Sentiment Analysis on Movie Reviews
Use Natural Language Processing (NLP) techniques like BERT, or TF-IDF with classifiers to analyze sentiment in movie reviews."	Dataset: IMDB Reviews
"2. Handwritten Digit Recognition
Implement Enseble Model to classify handwritten digits from the MNIST dataset."	Dataset: MNIST
"3. Email Spam Detection
Classify emails as spam or not using NLP techniques."	Dataset :  SPAM Detect
	
	
	
### Instruction	Eval Criteria
"1. Create a ML model by selecting any 1 of the above mentioned usecase. 
Students can choose any algorithm and approach to build it. "	REST API for best_model_parameter[method to be used GET], prediction[ method to be used POST] and training[ method to be used POST]
2. Use hyper-parameter tuning techniuque to find the best model and log all experiment run using ML-Flow	Experiment artifact should be visible in the code github repository
3. Create a Docker container for the Backed API and push it to docker hub	A working Docker Image of your application showcasing both the ML-flow UI and the REST API
	
	
### Guideline	
1. Students have to form a team of not more than 4 person. Single person team not recommended (have to take prior approval)	
2. Project team should use a single github repositroy for collaboaration	
"3. Each member should have a valid commit history of there contribution to the code repository. 
Any member who has no contribution will not be graded"	
4. Please do not copy code from ChatGPT	
5. AVOID pushing dataset to GitHub repo.	
