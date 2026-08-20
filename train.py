import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle


data = {
    'product': [
        # Electronics 20
        'iphone 15', 'iphone 14', 'samsung tv', 'dell laptop', 'boat earphones', 'apple watch',
        'sony headphones', 'hp printer', 'jbl speaker', 'playstation 5', 'ipad', 'macbook',
        'oneplus phone', 'mi tv', 'logitech mouse', 'keyboard', 'webcam', 'monitor', 'router', 'charger',
        
        # Clothing 20
        'nike shoes', 'adidas tshirt', 'levis jeans', 'puma shirt', 'zara dress',
        'h&m jacket', 'polo tshirt', 'reebok shoes', 'crocs', 'socks',
        'hoodie', 'jeans', 'kurta', 'saree', 'tshirt', 'shirt', 'pants', 'skirt', 'cap', 'belt',
        
        # Groceries 20
        'lays chips', 'milk', 'apple', 'bread', 'rice',
        'wheat flour', 'sugar', 'tea', 'coffee', 'butter',
        'banana', 'tomato', 'potato', 'onion', 'eggs', 'cheese', 'yogurt', 'oil', 'biscuits', 'chocolate'
    ],
    'category': ['electronics']*20 + ['clothing']*20 + ['groceries']*20
}

df = pd.DataFrame(data)
pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
pipeline.fit(df['product'], df['category'])
pickle.dump(pipeline, open('model.pkl', 'wb'))
print("Model retrained with 60 products!")