# Customer Segmentation with RFM

# 1. Business Problem
# 2. Understanding the Data
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Creating & Analysing RFM Segments


##########################################
# 1. Business Problem
##########################################

# An e-commerce company wants to segment its customers and determine marketing strategies according to these segments.

# The dataset named "Online Retail II" includes the sales of an UK-based online store between 01/12/2009 - 09/12/2011.

# dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Variables
# InvoiceNo     :Invoice number. The unique number of each transaction, namely the invoice. If it starts with C, it shows the canceled invoice.
# StockCode     :Unique number for each product.
# Description   :Product description
# Quantity      :It expresses how many of the products on the invoices have been sold.
# InvoiceDate   :Invoice date and time.
# UnitPrice     :Product price (in GBP)
# CustomerID    :Unique customer number
# Country       :Country where the customer lives.


##########################################
# 2. Understanding the Data
##########################################
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2009-2010')
df = df_.copy()


def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

cat_cols = [col for col in df.columns if df[col].dtypes in ['object', 'category', 'bool']]
num_cols = [col for col in df.columns if col not in cat_cols]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ['int64', 'float64']]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and df[col].dtypes in ['object', 'category']]
cat_cols = cat_cols + num_but_cat

df[cat_cols].nunique()


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * (dataframe[col_name].value_counts()) / len(dataframe)}))
    print('##################################')


for i in cat_cols:
    cat_summary(df, i)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(df.groupby([categorical_col]).agg({target: 'sum'}).sort_values(target, ascending=False).head())


for col in cat_cols:
    print(target_summary_with_cat(df, 'Quantity', col))

for col in cat_cols:
    print(target_summary_with_cat(df, 'Price', col))

df['TotalPrice'] = df['Quantity'] * df['Price']

df.groupby(['Invoice']).agg({'TotalPrice': 'sum'})


##########################################
# 3. Data Preparation
##########################################

df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.describe().T

df = df[~df['Invoice'].str.contains('C', na=False)]


##########################################
# 4. Calculating RFM Metrics
##########################################

# Recency, Frequency, Monetary

import datetime as dt

df.head()
df['InvoiceDate'].max()

today_date = dt.datetime(2010, 12, 11)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']
rfm.head()
rfm.describe().T

rfm = rfm[rfm['monetary'] > 0]

rfm.shape


##########################################
# 5. Calculating RFM Scores
##########################################

rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm['RFM_score'] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.head()


##########################################
# 6. Creating and Analysing RFM Segments
##########################################

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_score'].replace(seg_map, regex=True)

rfm[['segment', 'recency', 'frequency', 'monetary']].groupby('segment').agg(['mean', 'count'])

rfm[rfm['segment'] == 'champions'].head()
rfm[rfm['segment'] == 'champions'].index

new_df = pd.DataFrame()
new_df['champions_id'] = rfm[rfm['segment'] == 'champions'].index

new_df['champions_id'] = new_df['champions_id'].astype(int)

new_df.to_csv('champions.csv')

rfm.to_csv('rfm.csv')






