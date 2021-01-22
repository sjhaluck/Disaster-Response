import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    cat_counts = df.iloc[:,4:].apply(pd.value_counts).fillna(0).iloc[[1]].transpose(copy=True)
    cat_counts.rename(columns={1.0:'total'},inplace=True)
    cat_counts = cat_counts.sort_values(by='total',ascending=False)[:20]/df.shape[0]*100
    general_counts = cat_counts.iloc[:4]['total']
    general_names = list(general_counts.index)
    specific_counts = cat_counts.iloc[4:]['total']
    specific_names = list(specific_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=general_names,
                    y=general_counts
                )
            ],

            'layout': {
                'title': 'Distribution of General Message Classifications',
                'yaxis': {
                    'title': "Relative Frequency (% of total)"
                },
                'xaxis': {
                    'title': "General Message Classification"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=specific_names,
                    y=specific_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Specific Message Classifications',
                'yaxis': {
                    'title': "Relative Frequency (% of total)"
                },
                'xaxis': {
                    'title': "Specific Message Classification"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()