import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    loads the data fromn the two filepaths and merges them together
    
    Arguments:
    messages_filepath - the path to the messages CSV file
    categories_filepath - the path to the categories CSV file
    
    Returns:
    df - a dataframe of the messages and categories merged together
    '''
    
    # load the messages CSV file into a dataframe
    messages = pd.read_csv(messages_filepath)
    # load the categories CSV file into a dataframe
    categories = pd.read_csv(categories_filepath)
    # merge the categories with the messages on their shared id number
    df = messages.merge(categories,on=['id'])
    # return the merged dataframe
    return df


def clean_data(df):
    '''
    cleans the dataframe by separating out new identification columns
    and returning the dataframe without duplicates or null values
    
    Arguments:
    df - the version of the dataframe originally loaded from the data sources
    
    Returns:
    df - expanded dataframe with all labels properly separated
    '''
    
    # isolate the categories information and split it into 36 different columns
    categories = df['categories'].str.split(pat=';',expand=True)
    # isolate the first row of information to gather column names
    row = categories.iloc[0,:]
    # drop the last two characters from these names, so that it is only the identifier
    category_colnames = row.apply(lambda x: x[:-2])
    # rename all of the column names to the proper identifiers
    categories.columns = category_colnames
    # go through and format each column by taking only the numerical identifier from each value
    for column in categories:
        # isolate the numerical identifier from each string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert each number character into a numeric value
        categories[column] = pd.to_numeric(categories[column])
    # drop the previous categories column which was a long string
    df.drop(columns=['categories'],inplace=True)
    # concatenate the dataframe with the new category identifier dataframe
    df = pd.concat([df,categories],axis=1)
    # remove any duplicate values from the dataframe
    df.drop_duplicates(inplace=True)
    # remove any values that were not matched correctly with a message
    df.dropna(subset=['related'],inplace=True)
    # return the expanded and appropriately formatted dataframe
    return df
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace',index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()