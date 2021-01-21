# import all necessary packages
import sys
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt','wordnet','stopwords'])
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    load the information from the database
    
    Arguments:
    database_filepath - path to the SQLite database containing the information
    
    Returns:
    X - a dataframe containing the input messages for the model
    Y - a dataframe containing the 36 output classifiers for the model
    Y.columns - a list of the category labels
    '''
    
    # create the sqlalchemy engine to load the SQLite database
    engine = create_engine('sqlite:///'+database_filepath)
    # read the database into a dataframe
    df = pd.read_sql('DisasterResponse',engine)
    # create a dataframe of the messages to use as input to the model
    X = df[['message']].message
    # create a dataframe of the 36 classifiers as output for the model
    Y = df[df.columns[4:]]
    # return all relevant information
    return X, Y, Y.columns


def tokenize(text):
    '''
    break each message into tokens for processing in the model
    
    Arguments:
    text - a string of text to be parsed
    
    Returns:
    tokens - a list of made from the parsed words stripped to their roots
    '''
    
    # generate the list of common stop words in English
    stop_words = stopwords.words("english")
    # generate a list of words from the string of text
    words = word_tokenize(text.lower())
    # initialize the WordNetLemmatizer for identifying root words
    lemmatizer = WordNetLemmatizer()
    # create the list of root words from each work that is not classified as a common stop word
    tokens = [lemmatizer.lemmatize(word).strip() for word in words if word not in stop_words]
    # return the list of root words (tokens) to be process by the machine learning model
    return tokens


def build_model():    
    '''
    build the machine learning model into steps and parameters to test in order to optimize the model
    
    Arguments:
    None
    
    Returns:
    cv - the complete model prepared for training and cross-validation
    '''
    
    # construct a pipeline of steps to form the complete machine learning model
    # includes a vectorizer to transform each word into an accurate numerical model
    # a transformer to calculate and compare term frequencies and relative document frequencies
    # and a classifier to map the numerical models for the text to multiple output classifiers
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=10)))
    ])
    
    # create a dictionary of parameters to test in order to optimize the machine learning model
    parameters = {'vect__max_df':[0.8, 0.9],
                  'tfidf__smooth_idf':[True, False],
                  'clf__estimator':[RandomForestClassifier(n_estimators=10),
                                    RandomForestClassifier(n_estimators=20),
                                    KNeighborsClassifier(n_neighbors=11),
                                    KNeighborsClassifier(n_neighbors=17)]}

    # prepare the GridSearchCV model to optimize the pipeline for the model across all combinations of parameter
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1, cv=3, verbose=5)
    # return the completed model
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the trained machine learning model and output its statistics
    
    Arguments:
    model - the trained machine learning model
    X_test - the subset of inputs for testing the model
    Y_test - the subset of outputs for comparing the predictions of the model
    category_names - a list of the categories in the output of the model
    
    Returns:
    None
    '''
    
    # predict output values for the test inputs
    y_pred = model.predict(X_test)
    # correctly label the predicted outputs with the formal labels
    y_pred = pd.DataFrame(y_pred,columns=category_names)
    # for each label, print out a classificaiton report for analysis
    for col in Y_test.columns:
        print(col)
        print(classification_report(Y_test[col], y_pred[col]))
                         
def save_model(model, model_filepath):
    '''
    save the optimized machine learning model in a pickle file for later use
    
    Arguments:
    model - the trained machine learning model
    model_filepath - where to save the pickle file of the trained model
    
    Returns:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    # if the execution of the program contains enough arguments
    if len(sys.argv) == 3:
        # extract the path to the databse and the location to store the model
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # call the load_data function to extract input, output, and category names for the model
        X, Y, category_names = load_data(database_filepath)
        # split the data into a training set (80% of the data) and a testing set (20% of the data)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        # call the function to build the model
        model = build_model()
        
        print('Training model...')
        # train the model on the training set of data, which will optimize parameters with gridsearch
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        # evaluate the trained and optimized machine learning model
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        # save the model to the specified location as a pickle file
        save_model(model, model_filepath)

        print('Trained model saved!')

    # if the execution of the program did not include enough arguments
    else:
        # provide additional direction on how to execute the program correctly
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()