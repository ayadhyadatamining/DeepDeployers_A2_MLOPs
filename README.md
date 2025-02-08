# DeepDeployers_A2_MLOPs

**Sentiment Analysis Solution for Movie Reviews**
We'll implement two approaches (TF-IDF + Classifier and BERT) using Stanford's IMDB dataset1, then create a production-ready API system. The solution includes model development, API integration, and team workflow management.

System Architecture

text
graph TD
    A[User Interface] --> B(REST API)
    B --> C{Model Type}
    C --> D[TF-IDF + Classifier]
    C --> E[Fine-tuned BERT]
    B --> F[Database]
    F --> G[Model Parameters]
    F --> H[Training Logs]

Member 1-Arup Das : Data & Model Development
Dataset preprocessing and cleaning
Implement TF-IDF + classifier pipeline
BERT model fine-tuning
Model evaluation metrics

Member 2-Abhishek Prasad Nonia: API & Infrastructure
Design REST API endpoints
Implement request/response handling
Database integration for model parameters
Authentication system setup

Member 3-Piyush Sudhir Shobhane: Deployment & Testing
Docker containerization
CI/CD pipeline setup
Unit/Integration testing
Performance benchmarking

GitHub Collaboration Strategy
Branch Management

bash
main         - Production-ready code
development  - Stable development branch
feature/*    - Individual task branches
Equal Commit Distribution
Create issues for all subtasks
Use GitHub Projects for task tracking
Implement pair programming sessions for complex features
Follow the atomic commit strategy:
Small, focused commits
Conventional commit messages
Daily sync commits
Recommended Workflow

text
graph LR
    A[Create Issue] --> B[Assign to Member]
    B --> C[Create Feature Branch]
    C --> D[Develop & Commit]
    D --> E[Create PR]
    E --> F[Peer Review]
    F --> G[Merge to Development]
Key Considerations
Use model versioning (MLflow/DVC)
Implement API rate limiting
Set up automated testing (PyTest)
Use distributed training for BERT
Monitor model performance drift
This solution combines state-of-the-art NLP techniques with production-grade API design, following best practices for team collaboration1. The modular architecture allows easy swapping of model components while maintaining API consistency.


Problem Statement - Assignment 2

Submission Deadline - 26th Feb 2025	
	
ML model to develop	Dataset
"1. Sentiment Analysis on Movie Reviews
Use Natural Language Processing (NLP) techniques like BERT, or TF-IDF with classifiers to analyze sentiment in movie reviews."	Dataset: IMDB Reviews
"2. Handwritten Digit Recognition
Implement Enseble Model to classify handwritten digits from the MNIST dataset."	Dataset: MNIST
"3. Email Spam Detection
Classify emails as spam or not using NLP techniques."	Dataset :  SPAM Detect
	
	
	
Instruction	Eval Criteria
"1. Create a ML model by selecting any 1 of the above mentioned usecase. 
Students can choose any algorithm and approach to build it. "	REST API for best_model_parameter[method to be used GET], prediction[ method to be used POST] and training[ method to be used POST]
2. Use hyper-parameter tuning techniuque to find the best model and log all experiment run using ML-Flow	Experiment artifact should be visible in the code github repository
3. Create a Docker container for the Backed API and push it to docker hub	A working Docker Image of your application showcasing both the ML-flow UI and the REST API
	
	
Guideline	
1. Students have to form a team of not more than 4 person. Single person team not recommended (have to take prior approval)	
2. Project team should use a single github repositroy for collaboaration	
"3. Each member should have a valid commit history of there contribution to the code repository. 
Any member who has no contribution will not be graded"	
4. Please do not copy code from ChatGPT	
5. AVOID pushing dataset to GitHub repo.	
