import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message that needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the text provided
    """
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


@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    gen_counts = df.groupby('genre').count()['message']
    gen_names = list(gen_counts.index)
    
    # category data for plotting
    categories =  df[df.columns[4:]]
    catg_counts = (categories.mean()*categories.shape[0]).sort_values(ascending=False)
    catg_names = list(catg_counts.index)
    
    # Plotting of Categories Distribution in Direct Genre
    direct_catg = df[df.genre == 'direct']
    direct_catg_counts = (direct_catg.mean()*direct_catg.shape[0]).sort_values(ascending=False)
    direct_catg_names = list(direct_catg_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=gen_names,
                    y=gen_counts
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
        # category plotting (Visualization#2)
        {
            'data': [
                Bar(
                    x=catg_names,
                    y=catg_counts
                )
            ],

            'layout': {
                'title': 'Distribution of the Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
            
        },
        # Categories Distribution in Direct Genre (Visualization#3)
        {
            'data': [
                Bar(
                    x=direct_catg_names,
                    y=direct_catg_counts
                )
            ],

            'layout': {
                'title': 'Categories Distribution in the Direct Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in the Direct Genre"
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