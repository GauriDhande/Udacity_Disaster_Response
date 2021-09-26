# Udacity_Disaster_Response

File Structure


  •	app
  
  | - template
  
    |- master.html  main web app page 
    |- go.html  classification result page
  
  |- run.py   Flask file that runs app
  
  
  •	data
  
    |- disaster_categories.csv  Disaster Categories data
    
    |- disaster_messages.csv  Disaster Messages data
    
    |- process_data.py  Python script to process data
    
    |- DisasterResponse.db  database to save clean data to

  
  •	models
  
    |- train_classifier.py  python script to train the classifier

    |- classifier.pkl  saved model
  
  
  •	Preparation
  
    |- categories.csv  categories data set

    |- messages.csv  messages data set

    |- ETL_Preparation_Pipeline.ipynb  ETL Pipeline Jupyter Notebook

    |- ML Pipeline Preparation.ipynb  ML Pipeline Jupyter Notebook

    |- ETL_Preparation.db  ETL Database
  
  
  •	README.md
  
  
Instructions:


1.	Execute the following commands in the project's root directory to set up your database and model.
o	To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
o	To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/cv_AdaBoost.pkl
2.	Run the following command in the app's directory to run your web app. python run.py
3.	Go to http://0.0.0.0:3001/
