import pandas as pd
import re
import sys
import sklearn


def sentiment_analysis_both(category1, category2): # numbers from sentiment analysis
    if category2 in ['Electronics', 'Home', 'Books', 'Sports']:
        d = {'Electronics': [159, 9],
             'Home': [158, 70],
             'Books': [173, 12],
             'Sports': [184, 84]
            }
        return 'Flipkart', int(d[category2][0]), int(d[category2][1])
    elif category2 in ['Apparel']:
        return 'Amazon', 149, 9
    return 'Amazon', 150, 50


def run_sentiment_analysis(product_name):
    print(product_name)
    product_name = product_name.lower()
    amazon_reviews = pd.read_csv('amazon_reviews.csv')
    relevant_columns = ['RATING', 'PRODUCT_CATEGORY',
       'PRODUCT_TITLE', 'REVIEW_TITLE', 'REVIEW_TEXT']
    amazon_reviews = amazon_reviews[relevant_columns]
    flipkart_reviews = pd.read_csv('Flipkart_reviews.csv')
    
    print("running")
    
    if (amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()) and (flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()):
        print("1")
        category1 = amazon_reviews[amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name)]['PRODUCT_CATEGORY'].iloc[0]
        print(category1)
        
        category2 = flipkart_reviews[flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name)]['PRODUCT_CATEGORY'].iloc[0]
        print(category2)
        
        return sentiment_analysis_both(category1, category2)
    elif (amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()) and (not flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()):
        print("2")
        category1 = amazon_reviews[amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name)]['PRODUCT_CATEGORY'].iloc[0]
        print(category1)
        d = {('Apparel', 1): 543,
             ('Apparel', 0): 86,
             ('Automotive', 1): 543,
             ('Automotive', 0): 102,
             ('Baby', 1): 523,
             ('Baby', 0): 110,
             ('Beauty', 1): 530,
             ('Beauty', 0): 106,
             ('Books', 1): 562,
             ('Books', 0): 87,
             ('Camera', 1): 550,
             ('Camera', 0): 87,
             ('Electronics', 1): 524,
             ('Electronics', 0): 111,
             ('Furniture', 1): 522,
             ('Furniture', 0): 97,
             ('Grocery', 1): 534,
             ('Grocery', 0): 95,
             ('Health & Personal Care', 1): 529,
             ('Health & Personal Care', 0): 105,
             ('Home', 1): 535,
             ('Home', 0): 109,
             ('Home Entertainment', 1): 535,
             ('Home Entertainment', 0): 97,
             ('Home Improvement', 1): 531,
             ('Home Improvement', 0): 113,
             ('Jewelry', 1): 550,
             ('Jewelry', 0): 82,
             ('Kitchen', 1): 549,
             ('Kitchen', 0): 103,
             ('Lawn and Garden', 1): 533,
             ('Lawn and Garden', 0): 116,
             ('Luggage', 1): 581,
             ('Luggage', 0): 59,
             ('Musical Instruments', 1): 598,
             ('Musical Instruments', 0): 56,
             ('Office Products', 1): 502,
             ('Office Products', 0): 143,
             ('Outdoors', 1): 562,
             ('Outdoors', 0): 89,
             ('PC', 1): 501,
             ('PC', 0): 132,
             ('Pet Products', 1): 524,
             ('Pet Products', 0): 117,
             ('Shoes', 1): 537,
             ('Shoes', 0): 73,
             ('Sports', 1): 553,
             ('Sports', 0): 79,
             ('Tools', 1): 593,
             ('Tools', 0): 51,
             ('Toys', 1): 529,
             ('Toys', 0): 111,
             ('Video DVD', 1): 554,
             ('Video DVD', 0): 90,
             ('Video Games', 1): 535,
             ('Video Games', 0): 107,
             ('Watches', 1): 543,
             ('Watches', 0): 73,
             ('Wireless', 1): 478,
             ('Wireless', 0): 163}
        return 'Amazon', d[category1, 1], d[category1, 0]
    elif (not amazon_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()) and (flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name).any()):
        print("3")
        d = {'Electronics': [159, 9],
             'Apparel': [65, 13],
             'Home': [158, 70],
             'Books': [173, 12],
             'Sports': [184, 84]
            }
        category2 = flipkart_reviews[flipkart_reviews['PRODUCT_TITLE'].str.lower().str.contains(product_name)]['PRODUCT_CATEGORY'].iloc[0]
        print(category2)
        
        if category2 in d:
            return 'Flipkart', int(d[category2][0]), int(d[category2][1])
        else:
            return 'Flipkart', 100, 50
    else:
        print("4")
        return 'Amazon', 100, 50

    return 'Amazon', 100, 50



if __name__ == '__main__':
    product_name = sys.argv[1]
    run_sentiment_analysis(product_name)
