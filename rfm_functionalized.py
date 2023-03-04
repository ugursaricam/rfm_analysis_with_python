##########################################
# Functionalizing the Entire Process
##########################################

import pandas as pd
import numpy as np
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2009-2010')


def create_rfm(dataframe, csv=False):

    # Data Preparation
    dataframe['TotalPrice'] = dataframe['Quantity'] * dataframe['Price']
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains('C', na=False)]

    # Calculating RFM Metrics
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # Calculating RFM Scores
    rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    rfm['RFM_score'] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # Creating & Analysing RFM Segments
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
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
    rfm = rfm[['recency', 'frequency', 'monetary', 'segment']]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv('rfm.csv')

    return rfm

df = df_.copy()

rfm_new = create_rfm(df, csv=True)