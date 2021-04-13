import pandas as pd
import re
import sys
import sklearn
import pickle
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

stop=set(stopwords.words('english'))
print(stop)
nltk.download('stopwords')
#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#Removing the html strips
def strip_html(text):
    soup = bs(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

def combined_features(row):
    return row['PRODUCT_TITLE'] + ' '+ row['REVIEW_TITLE'] + ' ' + row['REVIEW_TEXT']

def run_deceptive_analysis(product_name):
    print(product_name)
    
    
    product_name = product_name.lower()
    amazon_reviews = pd.read_csv('amazon_reviews.csv')
    amazon_reviews.loc[amazon_reviews["LABEL"] == "__label1__", "LABEL"] = '1'
    amazon_reviews.loc[amazon_reviews["LABEL"] == "__label2__", "LABEL"] = '0'
    relevant_columns = ['RATING', 'PRODUCT_CATEGORY',
       'PRODUCT_TITLE', 'REVIEW_TITLE', 'REVIEW_TEXT', 'LABEL']
    amazon_reviews = amazon_reviews[relevant_columns]
    
    
    flipkart_reviews = pd.read_csv('Flipkart_reviews.csv')
    
    print("Flipkart processing")
    flipkart_reviews = flipkart_reviews.fillna("NA")
    for i in range(0,len(flipkart_reviews)-1):
        if type(flipkart_reviews.iloc[i]['REVIEW_TEXT']) != str:
            print(flipkart_reviews.iloc[i]['REVIEW_TEXT'])
            flipkart_reviews.iloc[i]['REVIEW_TEXT'] = str(flipkart_reviews.iloc[i]['REVIEW_TEXT'])
    
    
    
    print("running")
    
    if (amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()) and (flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()):
        print("1")
        amz = amazon_reviews[amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name)]
        amazon_reviews['REVIEW_TEXT']=amazon_reviews['REVIEW_TEXT'].apply(denoise_text)
        amazon_reviews['REVIEW_TEXT']=amazon_reviews['REVIEW_TEXT'].apply(remove_special_characters)
        amazon_reviews['REVIEW_TEXT']=amazon_reviews['REVIEW_TEXT'].apply(simple_stemmer)
        amazon_reviews['REVIEW_TEXT']=amazon_reviews['REVIEW_TEXT'].apply(remove_stopwords)

        amazon_reviews['ALL_FEATURES'] = amazon_reviews.apply(combined_features, axis=1)
        
        amz_honest = amz[amz['LABEL'] == '1']
        amz_deceptive = amz[amz['LABEL'] == '0']
        honest_amz_count = amz_honest.shape[0]
        deceptive_amz_count = amz_deceptive.shape[0]
        
        print(honest_amz_count)
        print(deceptive_amz_count)
        
        ratio_amz = honest_amz_count
        if deceptive_amz_count != 0:
            ratio_amz = honest_amz_count/deceptive_amz_count
        
        print("Amazon ratio ", ratio_amz)
        honest_percent_amz = (honest_amz_count/(amz.shape[0]))*100
        deceptive_percent_amz = (deceptive_amz_count/(amz.shape[0]))*100
        
        flip = flipkart_reviews[flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name)]
        flip['REVIEW_TEXT']=flip['REVIEW_TEXT'].apply(denoise_text)
        flip['REVIEW_TEXT']=flip['REVIEW_TEXT'].apply(remove_special_characters)
        flip['REVIEW_TEXT']=flip['REVIEW_TEXT'].apply(simple_stemmer)
        flip['REVIEW_TEXT']=flip['REVIEW_TEXT'].apply(remove_stopwords)

        flip['ALL_FEATURES'] = flip.apply(combined_features, axis=1)
        
        X_flipkart = flip['ALL_FEATURES']
        cv = CountVectorizer()
        ctmTr = cv.fit_transform(amazon_reviews['ALL_FEATURES'])
        X_test_dtm = cv.transform(X_flipkart)
        model = pickle.load(open('honest_deceptive.h5', 'rb'))
        
        X_Amazon = amazon_reviews['ALL_FEATURES']
        y_Amazon = amazon_reviews['LABEL']
        X_train_Amazon, X_test_Amazon, y_train_Amazon, y_test_Amazon = train_test_split(X_Amazon, y_Amazon, random_state=0)
        ctmTr = cv.fit_transform(X_train_Amazon)
        X_test_dtm = cv.transform(X_test_Amazon)
        
        y_pred_class = model.predict(X_test_dtm)
        
        unique, counts = np.unique(y_pred_class, return_counts=True)
        dict_flip = dict(zip(unique, counts))
        
        print(dict_flip)
        
        honest_flip_count = dict_flip['1']
        deceptive_flip_count = dict_flip['0']
        
        ratio_flip = honest_flip_count
        if deceptive_flip_count != 0:
            ratio_flip = honest_flip_count/deceptive_flip_count
        
        print("Flipkart ratio: ", ratio_flip)
        
        honest_percent_flip = (honest_flip_count/(flip.shape[0]))*100
        deceptive_percent_flip = (deceptive_flip_count/(flip.shape[0]))*100
        
        if ratio_amz > ratio_flip:
            print("1 Amazon")
            return 'Amazon', int(honest_percent_amz), int(deceptive_percent_amz)
        else:
            print("1 Flipkart")
            return 'Flipkart', int(honest_percent_flip), int(deceptive_percent_flip)
    elif (amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()) and (not flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()):
        print("2")
        amz = amazon_reviews[amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name)]
        amz['REVIEW_TEXT']=amz['REVIEW_TEXT'].apply(denoise_text)
        amz['REVIEW_TEXT']=amz['REVIEW_TEXT'].apply(remove_special_characters)
        amz['REVIEW_TEXT']=amz['REVIEW_TEXT'].apply(simple_stemmer)
        amz['REVIEW_TEXT']=amz['REVIEW_TEXT'].apply(remove_stopwords)

        amz['ALL_FEATURES'] = amz.apply(combined_features, axis=1)
        print(amz.columns)
        print(amz['LABEL'])
        amz_honest = amz[amz['LABEL'] == '1']
        amz_deceptive = amz[amz['LABEL'] == '0']
        honest_amz_count = amz_honest.shape[0]
        deceptive_amz_count = amz_deceptive.shape[0]
        
        print(honest_amz_count)
        print(deceptive_amz_count)
        
        ratio_amz = honest_amz_count
        if deceptive_amz_count != 0:
            ratio_amz = honest_amz_count/deceptive_amz_count
        
        print("Amazon ratio ", ratio_amz)
        honest_percent_amz = (honest_amz_count/(amz.shape[0]))*100
        deceptive_percent_amz = (deceptive_amz_count/(amz.shape[0]))*100
        print("2 Amazon")
        return 'Amazon', int(honest_percent_amz), int(deceptive_percent_amz)
    elif (not amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()) and (flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()):
        print("3")
        amazon_reviews['REVIEW_TEXT']=amazon_reviews['REVIEW_TEXT'].apply(denoise_text)
        amazon_reviews['REVIEW_TEXT']=amazon_reviews['REVIEW_TEXT'].apply(remove_special_characters)
        amazon_reviews['REVIEW_TEXT']=amazon_reviews['REVIEW_TEXT'].apply(simple_stemmer)
        amazon_reviews['REVIEW_TEXT']=amazon_reviews['REVIEW_TEXT'].apply(remove_stopwords)

        amazon_reviews['ALL_FEATURES'] = amazon_reviews.apply(combined_features, axis=1)
        
        flip = flipkart_reviews[flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name)]
        flip['REVIEW_TEXT']=flip['REVIEW_TEXT'].apply(denoise_text)
        flip['REVIEW_TEXT']=flip['REVIEW_TEXT'].apply(remove_special_characters)
        flip['REVIEW_TEXT']=flip['REVIEW_TEXT'].apply(simple_stemmer)
        flip['REVIEW_TEXT']=flip['REVIEW_TEXT'].apply(remove_stopwords)

        flip['ALL_FEATURES'] = flip.apply(combined_features, axis=1)
        
        X_flipkart = flip['ALL_FEATURES']
        cv = CountVectorizer()
        ctmTr = cv.fit_transform(amazon_reviews['ALL_FEATURES'])
        X_test_dtm = cv.transform(X_flipkart)
        model = pickle.load(open('honest_deceptive.h5', 'rb'))
        
        X_Amazon = amazon_reviews['ALL_FEATURES']
        y_Amazon = amazon_reviews['LABEL']
        X_train_Amazon, X_test_Amazon, y_train_Amazon, y_test_Amazon = train_test_split(X_Amazon, y_Amazon, random_state=0)
        ctmTr = cv.fit_transform(X_train_Amazon)
        X_test_dtm = cv.transform(X_test_Amazon)
        
        y_pred_class = model.predict(X_test_dtm)
        
        unique, counts = np.unique(y_pred_class, return_counts=True)
        dict_flip = dict(zip(unique, counts))
        
        print(dict_flip)
        
        honest_flip_count = dict_flip['1']
        deceptive_flip_count = dict_flip['0']
        
        ratio_flip = honest_flip_count
        if deceptive_flip_count != 0:
            ratio_flip = honest_flip_count/deceptive_flip_count
        
        print("Flipkart ratio: ", ratio_flip)
        
        honest_percent_flip = (honest_flip_count/(flip.shape[0]))*100
        deceptive_percent_flip = (deceptive_flip_count/(flip.shape[0]))*100
        print("3 Flipkart")
        return 'Flipkart', int(honest_percent_flip), int(deceptive_percent_flip)
    else:
        print("4 Amazon")
        return 'Amazon', 51, 49

    return 'Amazon', 51, 49



if __name__ == '__main__':
    product_name = sys.argv[1]
    run_deceptive_analysis(product_name)
