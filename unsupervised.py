import numpy as np
import pandas as pd
import datetime as dt

#read data
origin_data=pd.read_excel('C:\\Users\\tybty\\Downloads\\Online Retail.xlsx')
#origin_data.isna().sum()
origin_data= origin_data[~origin_data['InvoiceNo'].str.contains('C',na=False)]
origin_data['TotalPrice']=origin_data['Quantity']*origin_data['UnitPrice']
origin_data['InvoiceTime']=pd.DatetimeIndex(origin_data['InvoiceDate']).time
origin_data['InvoiceDate']=pd.DatetimeIndex(origin_data['InvoiceDate']).date
#print(origin_data['InvoiceDate'].max())
sd = dt.date(2011,12,9)
origin_data['Recency']=sd - origin_data['InvoiceDate']
origin_data['Recency'].astype('timedelta64[D]')
origin_data['Recency']=origin_data['Recency']/ np.timedelta64(1, 'D')
rfmTable = origin_data.groupby('CustomerID').agg({'Recency': lambda x:x.min(), # Recency
                                        'CustomerID': lambda x: len(x), # Frequency
                                        'TotalPrice': lambda x: x.sum()}) # Monetary Value
rfmTable.rename(columns={'CustomerID': 'Frequency',
                         'TotalPrice': 'Monetary'}, inplace=True)
rfmTable.head()
quartiles = rfmTable.quantile(q=[0.25,0.50,0.75])
print(quartiles, type(quartiles))
quartiles=quartiles.to_dict()
quartiles
## for Recency
def RClass(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4
## for Frequency and Monetary value
def FMClass(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1

rfmSeg = rfmTable
rfmSeg['R_Quartile'] = rfmSeg['Recency'].apply(RClass, args=('Recency',quartiles,))
rfmSeg['F_Quartile'] = rfmSeg['Frequency'].apply(FMClass, args=('Frequency',quartiles,))
rfmSeg['M_Quartile'] = rfmSeg['Monetary'].apply(FMClass, args=('Monetary',quartiles,))
#first approach
rfmSeg['RFMClass'] = rfmSeg.R_Quartile.map(str) + rfmSeg.F_Quartile.map(str) + rfmSeg.M_Quartile.map(str)
rfmSeg.groupby('RFMClass').agg('Monetary').mean()
rfmSeg.groupby('RFMClass').agg('Frequency').mean()
rfmSeg.groupby('RFMClass').agg('Recency').mean()

#second approach
rfmSeg['Total Score'] = rfmSeg['R_Quartile'] + rfmSeg['F_Quartile'] +rfmSeg['M_Quartile']
print(rfmSeg.head(), rfmSeg.info())

rfmSeg.groupby('Total Score').agg('Monetary').mean()
rfmSeg.groupby('Total Score').agg('Monetary').mean().plot(kind='bar', colormap='Blues_r')
rfmSeg.groupby('Total Score').agg('Frequency').mean()
rfmSeg.groupby('Total Score').agg('Frequency').mean().plot(kind='bar', colormap='Blues_r')
rfmSeg.groupby('Total Score').agg('Recency').mean()
rfmSeg.groupby('Total Score').agg('Recency').mean().plot(kind='bar', colormap='Blues_r')

rfmSeg.sort_values(by=['RFMClass', 'Monetary'], ascending=[True, False]).to_csv('C:\\Users\\tybty\\Downloads\\Output.csv')
#rfmSeg.sort_values(by=['Total Score', 'Monetary'], ascending=[True, False]).to_csv('C:\\Users\\tybty\\Downloads\\Output.csv')



