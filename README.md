# HateSpeechAnalysis

This project was created with the help of AI.  
It implements a **Python model for detecting offensive (hate speech) messages**, mainly trained on Twitter data.  

The model is first trained using the provided dataset (`train.csv`), and then it can be tested using the additional prediction scripts.  
From testing, the model appears to be **overfitted**, possibly due to the imbalance between offensive (~16,000 samples) and neutral/inciting tweets (~8,000 total).

---

## Files

- **antrenareModel1.py** – trains the classification model using `train.csv`  
- **predictie.py** – runs predictions using the trained model  
- **predictie_manual.py** – allows manual input for testing predictions  
- **preluareDateTw.py** – collects tweets (requires Twitter API credentials)  
- **train.csv** – main dataset for training  
- **date_noi_tweeter.csv** – additional dataset for testing  
- **model_clasificare.joblib** – saved trained model  
- **vectorizator.joblib** – saved text vectorizer (used for preprocessing)  


