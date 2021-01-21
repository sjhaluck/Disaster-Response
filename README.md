# Disaster Response Pipeline Project

## Summary
This project uses a machine learning model to classify disaster/emergency messages into different categories. This project could serve to improve the effectiveness and efficiency of emergency response organizations when a disaster occurs.

The project includes:
- code to gather and clean the data
- code to build, train, and package the machine learning model
- code to deploy a webapp interface

### Sources:
- Udacity Data Science Nanodegree
- FigureEight

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:
data/
-- disaster_messages.csv
-- disaster_categories.csv
-- process_data.py

models/
-- train_classifier.py

app/
-- run.py
-- templates/
---- go.html
---- master.html