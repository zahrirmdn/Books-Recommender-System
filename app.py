from flask import Flask, render_template, request  # type: ignore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

app = Flask(__name__)

# Constants
URL = r'C:\Users\zahri\Downloads\books-recommender-system-main\books-recommender-system-main\google_books_1299.csv'

def clean_data(data):
    '''Clean and preprocess the data'''
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data = data.drop_duplicates(keep=False)
    data = data.drop_duplicates(['title'], keep='first')
    data['description'] = data['description'].fillna('')
    data['generes'] = data['generes'].fillna('')
    return data

def read_data(path):
    '''Read and preprocess data'''
    try:
        data = pd.read_csv(path)
        df = clean_data(data)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the data: {e}")

def compute_cosine_similarity_matrix(data, method='tfidf'):
    '''Compute the cosine similarity matrix based on the chosen method'''
    if method not in ['tfidf', 'count']:
        raise ValueError("Invalid method. Choose either 'tfidf' or 'count'.")
    
    vectorizer = TfidfVectorizer(stop_words='english') if method == 'tfidf' else CountVectorizer(stop_words='english')
    try:
        matrix = vectorizer.fit_transform(data)
    except Exception as e:
        raise ValueError(f"Error during vectorization: {e}")

    cosine_sim_matrix = linear_kernel(matrix, matrix) if method == 'tfidf' else cosine_similarity(matrix, matrix)
    return cosine_sim_matrix

def get_recommendations(title, method='tfidf'):
    '''Get book recommendations based on the chosen method'''
    df = read_data(URL)
    
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    if title not in indices:
        return None, f"Book title '{title}' not found in dataset."
    
    # Use the appropriate column based on the method
    if method == 'tfidf':
        cosine_sim_matrix = compute_cosine_similarity_matrix(df['description'], method='tfidf')
    elif method == 'count':
        cosine_sim_matrix = compute_cosine_similarity_matrix(df['generes'], method='count')
    else:
        raise ValueError("Invalid method. Please choose 'tfidf' or 'count'.")
    
    index = indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices], None

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations_tfidf = []
    recommendations_count = []
    error_message = None

    if request.method == "POST":
        title = request.form.get("book_title")
        recommendations_tfidf, error_message = get_recommendations(title, method='tfidf')
        if not error_message:
            recommendations_count, _ = get_recommendations(title, method='count')

    # Convert recommendations to list if they're not empty
    recommendations_tfidf = recommendations_tfidf.tolist() if isinstance(recommendations_tfidf, pd.Series) else recommendations_tfidf
    recommendations_count = recommendations_count.tolist() if isinstance(recommendations_count, pd.Series) else recommendations_count

    return render_template('index.html', recommendations_tfidf=recommendations_tfidf, recommendations_count=recommendations_count, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
