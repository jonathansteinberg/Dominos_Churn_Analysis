#!pip install sas7bdat

import pandas as pd
import numpy as np
import random

import seaborn as sns
import matplotlib.pyplot as plt

import time
import datetime
from datetime import datetime

import statistics as stat
import pyarrow as pa
import pyarrow.parquet as pq
from sas7bdat import SAS7BDAT
import platform
import operator as op

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import statsmodels.api
from sklearn import metrics

print('Python: ', platform.python_version())
print('pandas: ', pd.__version__)
print('pyarrow: ', pa.__version__)

with SAS7BDAT('/Users/jonathansteinberg/Desktop/Dominos SAS/transactions1.sas7bdat', skip_header=False) as reader:
    df_full = reader.to_data_frame()

df_sample = df_full
df_sample = df_sample.sort_values(by="DateOfOrder") 

print("########## Data Summary ##########")
print("Number of Customers:",len(df_sample.AddressId.unique()))
print("Number of Observations:",len(df_sample)) 
print("Start Date:",list(df_sample.DateOfOrder)[0]) 
print("End Date:",list(df_sample.DateOfOrder)[-1])

######################################
########## Data Preperation ##########
######################################

### Make Timestamp Colum
timestamps = []
for date,time in zip(list(df_sample.DateOfOrder), list(df_sample.OrderTime)):
    import datetime 
    combined_time = datetime.datetime.combine(date, time)
    from datetime import datetime
    timestamp_i = datetime.timestamp(combined_time)
    timestamps.append(timestamp_i)
df_sample['TimestampOfOrder'] = timestamps
timestamp_Feb_1 = 1454302800
timestamp_Mar_1 = 1456808400
df_sample = df_sample.sort_values(by="TimestampOfOrder")

### Make Profit Column
df_sample['Profit'] = df_sample['OrderAmount']-df_sample['IdealFoodCost']

###################################
########## Predict Churn ##########
###################################
 
### Create Monthly Datasets
import datetime
df_sample_Jan = df_sample[df_sample.DateOfOrder < datetime.date(2016, 2, 1)]  
df_sample_Feb = df_sample[(df_sample.DateOfOrder >= datetime.date(2016, 2, 1))&(df_sample.DateOfOrder < datetime.date(2016, 3, 1))]  
df_sample_Mar = df_sample[df_sample.DateOfOrder >= datetime.date(2016, 3, 1)]  
df_sample_Jan_customers = list(df_sample_Jan.AddressId.unique())
df_sample_Feb_customers = list(df_sample_Feb.AddressId.unique())
df_sample_Mar_customers = list(df_sample_Mar.AddressId.unique())

print("########## Monthly Breakdown ##########")
print("Amount of orders in January:",len(df_sample_Jan))
print("Amount of orders in February:",len(df_sample_Feb))
print("Amount of orders in March:",len(df_sample_Mar))
print("Amount of unique customers in January:",len(df_sample_Jan_customers))
print("Amount of unique customers in February:",len(df_sample_Feb_customers))
print("Amount of unique customers in March:",len(df_sample_Mar_customers))
print("Percent of total orders in January:",len(df_sample_Jan)/len(df_sample))
print("Percent of total orders in February:",len(df_sample_Feb)/len(df_sample))
print("Percent of total orders in March:",len(df_sample_Mar)/len(df_sample))

### RMF January
list_AddressId_i_Jan = []
for AddressId in df_sample_Jan.AddressId.unique():
    # Unique AddressId 
    AddressId_i = df_sample_Jan[df_sample_Jan.AddressId==AddressId]
    # Frequency (Orders per Month)
    frequency = len(AddressId_i)  
    # Recency 
    recency = (timestamp_Feb_1-list(AddressId_i.TimestampOfOrder)[-1])/86400
    # Monetary (Customer Value)
    monetary = AddressId_i.OrderAmount.sum()
    # Customer Lifespan
    customer_lifespan = (list(AddressId_i.TimestampOfOrder)[-1]-list(AddressId_i.TimestampOfOrder)[0])/86400
    # Profit 
    profit = AddressId_i.Profit.sum()
    # Average Profit per Order 
    coupons_used = len(AddressId_i[(AddressId_i.CouponCode != '')&(AddressId_i.DiscountAmount > 0)])
    # Discount Amount
    discount_amount = AddressId_i.DiscountAmount.sum()
    
    dict_AddressId_i_Jan = {"AddressId": AddressId,
                            "Recency_Jan": recency,
                            "Frequency_Jan": frequency,
                            "Monetary_Jan": monetary,
                            "CustomerLifespan_Jan": customer_lifespan,
                            "Profit_Jan": profit,
                            "CouponsUsed_Jan": coupons_used,
                            "DiscountAmount_Jan": discount_amount
                            }
    
    list_AddressId_i_Jan.append(dict_AddressId_i_Jan)
df_sample_AddressId_i_Jan = pd.DataFrame(list_AddressId_i_Jan)

### RMF Feburary
list_AddressId_i_Feb = []
for AddressId in df_sample_Feb.AddressId.unique():
    # Unique AddressId 
    AddressId_i = df_sample_Feb[df_sample_Feb.AddressId==AddressId]
    # Frequency 
    frequency = len(AddressId_i)  
    # Recency
    recency = (timestamp_Mar_1-list(AddressId_i.TimestampOfOrder)[-1])/86400
    # Monetary (Purchase Value)
    monetary = AddressId_i.OrderAmount.sum()
    # Customer Lifespan
    customer_lifespan = (list(AddressId_i.TimestampOfOrder)[-1]-list(AddressId_i.TimestampOfOrder)[0])/86400
    # Profit 
    profit = AddressId_i.Profit.sum()
    # Average Profit per Order 
    coupons_used = len(AddressId_i[(AddressId_i.CouponCode != '')&(AddressId_i.DiscountAmount > 0)])
    # Percent of Orders Coupon Used
    discount_amount = AddressId_i.DiscountAmount.sum()
    
    dict_AddressId_i_Feb = {"AddressId": AddressId,
                            "Recency_Feb": recency,
                            "Frequency_Feb": frequency,
                            "Monetary_Feb": monetary,
                            "CustomerLifespan_Feb": customer_lifespan,
                            "Profit_Feb": profit,
                            "CouponsUsed_Feb": coupons_used,
                            "DiscountAmount_Feb": discount_amount
                            }
    
    list_AddressId_i_Feb.append(dict_AddressId_i_Feb)
df_sample_AddressId_i_Feb = pd.DataFrame(list_AddressId_i_Feb)

### Churned January
churned_binary_Jan = []
for AddressId in df_sample_Jan.AddressId.unique():
    if (AddressId in df_sample_Jan_customers) and (AddressId not in df_sample_Feb_customers):
        churned_binary_Jan.append(1)
    else:
        churned_binary_Jan.append(0)
df_churned_Jan = pd.DataFrame()
df_churned_Jan['AddressId'] = df_sample_Jan.AddressId.unique()
df_churned_Jan['Churned_Feb'] = churned_binary_Jan
df_churned_Jan = pd.merge(df_sample_AddressId_i_Jan, df_churned_Jan, how="left", on=["AddressId"])

### Churned Feburary
churned_binary_Feb = []
for AddressId in df_sample_Feb.AddressId.unique():
    if (AddressId in df_sample_Feb_customers) and (AddressId not in df_sample_Mar_customers):
        churned_binary_Feb.append(1)
    else:
        churned_binary_Feb.append(0)     
df_churned_Feb = pd.DataFrame()
df_churned_Feb['AddressId'] = df_sample_Feb.AddressId.unique()
df_churned_Feb['Churned_Mar'] = churned_binary_Feb
df_churned_Feb = pd.merge(df_sample_AddressId_i_Feb, df_churned_Feb, how="left", on=["AddressId"])

### Churn Prediction DataFrame
df_churn_prediction = df_churned_Jan.append(df_churned_Feb)

### Stats for Problem Section of Report
print("########## Problem Section Stats ##########")
print("Number of Customers Churned:",df_churn_prediction.Churned_Feb.sum()+df_churn_prediction.Churned_Mar.sum())
print("Perecnt of Customers Churned:",(df_churn_prediction.Churned_Feb.sum()+df_churn_prediction.Churned_Mar.sum())/len(df_churn_prediction))
print("Number of Customers that made one order in Jan and Feb:", len(df_churn_prediction[df_churn_prediction.Frequency_Jan == 1])+len(df_churn_prediction[df_churn_prediction.Frequency_Feb == 1]))
print("Number of Churned Customers that made one order in Jan and Feb:",(len(df_churn_prediction[(df_churn_prediction.Frequency_Jan == 1)&(df_churn_prediction.Churned_Feb == 1)])+len(df_churn_prediction[(df_churn_prediction.Frequency_Feb == 1)&(df_churn_prediction.Churned_Mar == 1)]))/(df_churn_prediction.Churned_Feb.sum()+df_churn_prediction.Churned_Mar.sum()))
#^ Customers that made one order in Jan and Feb/churned customers 

### Predict Churn
X_train = df_churn_prediction[['Recency_Jan','Frequency_Jan','Monetary_Jan']].dropna()
y_train = df_churn_prediction["Churned_Feb"].dropna()

X_test = df_churn_prediction[['Recency_Feb','Frequency_Feb','Monetary_Feb']].dropna()
y_test = df_churn_prediction["Churned_Mar"].dropna()

logit = LogisticRegression()
model = logit.fit(X_train,y_train)
y_test_pred = model.predict(X_test)
y_test_pred_prob = model.predict_proba(X_test)

print("########## Prediction Evaluation ##########")
print("Prediction Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("Prediction Precision:",metrics.precision_score(y_test, y_test_pred))
print("Prediction Recall:",metrics.recall_score(y_test, y_test_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_test_pred))
print("Model Intercept:",model.intercept_)
print("Model Coefficients (R,F,M):",model.coef_)

df_churned_Feb['Churned_Mar_Pred'] = y_test_pred
df_churned_Feb['Churned_Mar_Pred_Prob'] = pd.DataFrame(y_test_pred_prob).iloc[:,1]

##############################################
########## Analysis of Alternatives ##########
##############################################

df_alt = df_churned_Feb
df_alt.AverageDiscountAmount_Feb = df_alt.DiscountAmount_Feb.replace(np.nan, 0)
number_of_customers_Feb = len(df_alt)
df_alt.columns = ["AddressId","Recency","Frequency","Monetary","Lifespan","Profit","CouponsUsage","DiscountAmount","ActualChrunRate","PredictedChurnRate","ChurnProbability"]
    
########## Alternative 1: Offer coupons to all customers predicted to churn 
df_alt1_churned_customers = df_alt[df_alt.PredictedChurnRate == 1]
size_of_group_alt1 = len(df_alt1_churned_customers)
df_alt1_churned_customers = df_alt1_churned_customers.iloc[:,1:10].mean()
df_alt1_churned_customers = pd.DataFrame(df_alt1_churned_customers).transpose()
df_alt1_churned_customers.index = ["Predcited to Churn"]
df_alt1_churned_customers.insert(0,"NumberOfCustomers", size_of_group_alt1) ### 1ST ROW ###

########## Alternative 2: Offer coupons to lowest tier of recency, frequency, monetary 

### Clustering 
n = 4
X = df_alt[["Recency", "Frequency", "Monetary"]]
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n, random_state=0)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

df_alt_new = df_alt
df_alt_new['ClusterLabel'] = labels
X = df_alt_new[["Recency", "Frequency", "Monetary"]]
X["Recency"] = 30-X["Recency"]
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
df_alt_new = df_alt_new.join(pd.DataFrame(X_std, columns = ["Recency_STD", "Frequency_STD", "Monetary_STD"]))
df_alt_new['CompoundScore'] = ((1/3)*df_alt_new.Recency_STD)+((1/3)*df_alt_new.Frequency_STD)+((1/3)*df_alt_new.Monetary_STD)

# Scatter Plot: Recency vs Frequency
sns.scatterplot(
    data=df_alt_new, 
    x='Recency', 
    y='Frequency', 
    hue='ClusterLabel',
    )
plt.title("Recency vs Frequency (n="+str(n)+")")
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot: Recency vs Monetary
sns.scatterplot(
    data=df_alt_new, 
    x='Recency', 
    y='Monetary', 
    hue='ClusterLabel',
    )
plt.title("Recency vs Monetary (n="+str(n)+")")
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.show()

# Scatter Plot: Frequency vs Monetary
sns.scatterplot(
    data=df_alt_new, 
    x='Frequency', 
    y='Monetary', 
    hue='ClusterLabel',
    )
plt.title("Frequency vs Monetary (n="+str(n)+")")
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.show()

print(df_alt_new[['ClusterLabel','CompoundScore']].groupby('ClusterLabel').mean())


df_alt2_lowest_rfm = df_alt_new[df_alt_new.ClusterLabel == 0]
size_of_group_alt2 = len(df_alt2_lowest_rfm)
df_alt2_lowest_rfm = df_alt2_lowest_rfm.iloc[:,1:10].mean()
df_alt2_lowest_rfm = pd.DataFrame(df_alt2_lowest_rfm).transpose()
df_alt2_lowest_rfm.index = ["Lowest RFM #1"]
df_alt2_lowest_rfm.insert(0,"NumberOfCustomers", size_of_group_alt2) ### 2ND ROW ###

########## Alternative 5: Other Lowest RFM group
df_alt5_lowest_rfm = df_alt_new[df_alt_new.ClusterLabel == 3]
size_of_group_alt5 = len(df_alt5_lowest_rfm)
df_alt5_lowest_rfm = df_alt5_lowest_rfm.iloc[:,1:10].mean()
df_alt5_lowest_rfm = pd.DataFrame(df_alt5_lowest_rfm).transpose()
df_alt5_lowest_rfm.index = ["Lowest RFM #2"]
df_alt5_lowest_rfm.insert(0,"NumberOfCustomers", size_of_group_alt5) ### 2ND ROW ###

########## Alternative 3: Offer coupons to customers who have a high risk of churn 
churn_75th_percentile = df_alt.ChurnProbability.describe()[6]
df_alt3_high_risk_churn = df_alt[df_alt.ChurnProbability > churn_75th_percentile]
size_of_group_alt3 = len(df_alt3_high_risk_churn)
df_alt3_high_risk_churn = df_alt3_high_risk_churn.iloc[:,1:10].mean()
df_alt3_high_risk_churn = pd.DataFrame(df_alt3_high_risk_churn).transpose()
df_alt3_high_risk_churn.index = ["High Risk Churn"]
df_alt3_high_risk_churn.insert(0,"NumberOfCustomers", size_of_group_alt3) ### 3RD ROW ###

########## Alternative 4: Offer coupons to random subset of customers 

list_sample_alt_4 = random.sample(list(df_alt.AddressId.unique()),int(stat.mean([size_of_group_alt1,size_of_group_alt2,size_of_group_alt3,size_of_group_alt5])))
df_alt4_random  = df_alt[df_alt.AddressId.isin(list_sample_alt_4)]
size_of_group_alt4 = len(df_alt4_random)
df_alt4_random = df_alt4_random.iloc[:,1:10].mean()
df_alt4_random = pd.DataFrame(df_alt4_random).transpose()
df_alt4_random.index = ["Random Subset"]
df_alt4_random.insert(0,"NumberOfCustomers", size_of_group_alt4) ### 4th ROW ###

df_final = df_alt1_churned_customers.append(df_alt3_high_risk_churn).append(df_alt2_lowest_rfm).append(df_alt5_lowest_rfm).append(df_alt4_random)

############################################################
########## Recommendation: Which Coupons to Offer ##########
############################################################

### Chosen Altnernative: High Risk Churn Customers 
df_x = df_alt[df_alt.ChurnProbability > churn_75th_percentile]
df_xx = df_sample_Feb[df_sample_Feb.AddressId.isin(df_x.AddressId)]

### Place: Where to send coupons 
print("Amount of high risk churn customers that order via website:",df_xx['OrderType'].value_counts()[0])
print("Amount of high risk churn customers that order via phone:",df_xx['OrderType'].value_counts()[1])
print("Amount of high risk churn customers that order via walk-in:",df_xx['OrderType'].value_counts()[2])

### Product: Type of coupons

# Unpopular Foods
print("High risk churn customers that did not order wings:",len(df_xx[(df_xx.BoneInCount == 0)&(df_xx.BoneInCount == 0)]))
print("High risk churn customers that did not order drink:",len(df_xx[(df_xx.Drink12ozCount == 0)&(df_xx.Drink20ozCount == 0)]))
print("High risk churn customers that did not order dessert:",len(df_xx[(df_xx.DessertCount == 0)]))

# Popular Coupons for Unpopular Foods
df_xx = df_sample_Feb[(df_sample_Feb['CouponDesc'].str.contains("Wings"))|(df_sample_Feb['CouponDesc'].str.contains("WINGS"))|(df_sample_Feb['CouponDesc'].str.contains("wings"))|(df_sample_Feb['CouponDesc'].str.contains("bone"))|(df_sample_Feb['CouponDesc'].str.contains("BONE"))|(df_sample_Feb['CouponDesc'].str.contains("Bone"))]
print("Most popular coupon including wings:\n",pd.DataFrame(df_xx['CouponDesc'].value_counts()).iloc[0,:])

df_xx = df_sample_Feb[(df_sample_Feb['CouponDesc'].str.contains("DRINK"))|df_sample_Feb['CouponDesc'].str.contains("Drink")|df_sample_Feb['CouponDesc'].str.contains("drink")]
print("Most popular coupon including drink:\n",pd.DataFrame(df_xx['CouponDesc'].value_counts()).iloc[1,:])
# ^Most popular one did not have a price associated with it 

df_xx = df_sample_Feb[(df_sample_Feb['CouponDesc'].str.contains("lava"))|(df_sample_Feb['CouponDesc'].str.contains("Lava"))|(df_sample_Feb['CouponDesc'].str.contains("LAVA"))|(df_sample_Feb['CouponDesc'].str.contains("cina"))|(df_sample_Feb['CouponDesc'].str.contains("Cina"))|(df_sample_Feb['CouponDesc'].str.contains("CINA"))]
print("Most popular coupon including desert:\n",pd.DataFrame(df_xx['CouponDesc'].value_counts()).iloc[0,:])





















