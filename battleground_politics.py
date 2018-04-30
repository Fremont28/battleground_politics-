#4/23/18
#forecasting the 2016 presidential election in PA, NC, and FLA 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import time
import sklearn as sk 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import pydotplus  
from IPython.display import Image

polls=pd.read_csv("upshot-siena-polls.csv")
polls.info()
polls['state'].unique() 

#state labels 
state_code={'Pennsylvania':0,
'Florida':1,'North Carolina':2}
polls['state_code']=polls['state'].replace(state_code)
#gender
gender_code={'Male':0,'Female':1}
polls['gender_code']=polls['gender'].replace(gender_code)
#senator
sen_fav={'Richard Burr, the Republican':0,
"[DO NOT READ] Don't know/No opinion":1,
"[DO NOT READ] Won't vote":2,
"Deborah Ross, the Democrat":3,
"[DO NOT READ] Someone else (specify)":4
}
polls['sen_fav']=polls['vt_sen'].replace(sen_fav) 
#governor 
gov_fav={"Pat McCrory, the Republican":0,
"Roy Cooper, the Democrat":1,
"[DO NOT READ] Don't know/No opinion":2,
"[DO NOT READ] Won't vote":3,"[DO NOT READ] Someone else (specify)":4} 
polls['gov_fav']=polls['vt_gov'].replace(gov_fav) 
#vote likelihood
vote_lh={"Ten (definitely will vote)":10,
"Nine":9,"Three":3,"One (definitely will NOT vote)":1,
"Eight":8,"Five":5,"Two":2,"Six":6,'Seven':7,
"Four":4,"[DO NOT READ] Don't know/No opinion":5,"nan":5}
polls['vote_lh']=polls['scale'].replace(vote_lh)
#party id
party_id={'Republican':0,
'Democrat':1,'Independent (No party)':2,
'[DO NOT READ] Refused':3,'or as a member of another political party':4} 
polls['party']=polls['partyid'].replace(party_id)
#education
education={'Graduate or Professional degree':1,
'Some college or trade school':1,"Bachelors' degree":1,
'High school':0,'[DO NOT READ] Refused':2,
'Grade school':1}
polls['education']=polls['educ'].replace(education)
#race
race_card={'Caucasian/White':0,
'[DO NOT READ] Other/Something else (specify)':1,
'African American/Black':2,'[DO NOT READ] Refused':3,
'Asian':4}
polls['race_card']=polls['race'].replace(race_card)
#presidential fav (for prediction purposes)
prez_fav={'Donald Trump, the Republican':0,
'Hillary Clinton, the Democrat':1,
"[DO NOT READ] Won't vote ":2,
"[DO NOT READ] Don't know/No opinion":3,
"[DO NOT READ] Someone else (specify)":4,
"nan":5} 
polls['pres_fav']=polls['vt_pres_2'].replace(prez_fav)
#democrat view
vote_dem={'Unfavorable':0,
'Favorable':1,"[DO NOT READ] Don't know/No opinion":2}
polls['dem_fav']=polls['d_pres_fav'].replace(vote_dem)
#republican view
vote_rep={'Unfavorable':0,
'Favorable':1,"[DO NOT READ] Don't know/No opinion":2}
polls['rep_fav']=polls['r_pres_fav'].replace(vote_rep)

#subset poll data 
polls1=polls[(polls['pres_fav']==0) | (polls['pres_fav']==1)]
polls1.shape #3359,45 
polls_x=polls 
polls_x['gender_coded']=np.where(polls_x['gender']=="Female",1,0)
polls_x['vote_president']=np.where(polls_x['vt_pres_2']=="Donald Trump, the Republican",0,
np.where(polls_x['vt_pres_2']=="Hillary Clinton, the Democrat",1)) ###???

#find missing values  
limpiar_data.isnull().T.any().T.sum() #rows with nan
nan_rows=limpiar_data[limpiar_data.isnull().T.any().T] #source: https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
limpiar_data=limpiar_data.dropna(how='all')
limpiar_data.to_csv("orange_sprite.csv")
#clean the data for nan 
limpiar_data=polls1[['gender_code','vote_lh','party',
'education','race_card','dem_fav','rep_fav','pres_fav']]
limpiar_data1=limpiar_data.dropna(axis=1,how='all')
limpiar_data1=limpiar_data1.apply(lambda x:x.fillna(x.mean()),axis=0)

#split dataset into train and test features 
X_train,X_test=train_test_split(limpiar_data1,test_size=0.35, random_state=792)
#check for missing values  
nan_rows1=limpiar_data1[limpiar_data1.isnull().T.any().T]
#create the nb classifier
mnb=MultinomialNB()
features=['gender_code','vote_lh','party',
'education','race_card','dem_fav','rep_fav']

#train the classifier
mnb.fit(X_train[features].values,
X_train["pres_fav"])
#
vote_predictions=mnb.predict(X_test[features])
vote_predictions
#accuracy
X_test.shape[0]
X_test["pres_fav"]!=vote_predictions.sum()
100*(1-X_test["pres_fav"]!=vote_predictions).sum()/ X_test.shape[0]
(X_test.shape[0],(X_test["pres_fav"]!=vote_predictions).sum(),
100*(1-X_test["pres_fav"] !=vote_predictions).sum()/X_test.shape[0]) #90.93% accuracy?

#sample obervations about presidential favorites
mean_clinton=np.mean(X_train["pres_fav"]) #0.5396 (nod to h. clinton)
mean_trump=1-mean_clinton #46.04
#some obs. sobre about clinton and trump female supporters
mean_female=np.mean(X_train[X_train["gender_code"]==1]["pres_fav"]) #58.7%
std_female=np.std(X_train[X_train["gender_code"]==1]["pres_fav"]) #0.49

mean_male=np.mean(X_train[X_train["gender_code"]==0]["pres_fav"])
#white male voters
w_male=X_train[(X_train["gender_code"]==0 )& (X_train["race_card"]==2)]
mean_w_male=np.mean(w_male)
mean_w_male
w_female=X_train[(X_train["gender_code"]==1) & (X_train["race_card"]==2)]
mean_w_female=np.mean(w_female)
mean_w_female
#decison trees
limpiar_data1.info() 
limpiar_data1.pres_fav.value_counts(normalize=True).plot(kind="bar",title="Clinton Edges Out Trump In Key Battleground States")
plt.show() 

#split the data
data_tree=limpiar_data1.values #1-clinton, 0-trump 
train=data_tree[0:2400]
test=data_tree[2401:3559]
x_train=train[:,0:7] 
y_train=train[:,7]
x_test=test[:,0:7]
y_test=test[:,7]

limpiar_data1.info() 
tree1=DecisionTreeClassifier(max_depth=3,random_state=732)
tree1.fit(x_train,y_train)
p1=tree1.predict_proba(x_test)[:,1]
p1.mean() #52.8 clinton probability that she wins FLA, VA, and PA
roc_auc_score(y_test,p1) #0.9867 
print(tree1,x_train) 

#simplify the model
drop=["gender_code"]
x_train1=np.delete(x_train[:,1],None)

#random forest
#bootsrapped averaging (bagging) and random forest when used with decision trees
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=5,max_features=4)
rf.fit(x_train,y_train)
p2=rf.predict_proba(x_test)[:,1]
p2.mean() #52.2% clinton edge 
roc_auc_score(y_test,p2) #96.8%

#decision tree 
clf_tree=DecisionTreeClassifier(max_depth=None,min_samples_split=3,random_state=0)
scores_tree=cross_val_score(clf_tree,x_train,y_train)
scores_tree.mean() #94.4%

clf_rf=RandomForestClassifier(n_estimators=5,max_depth=None,
min_samples_split=2)
clf_rf.fit(x_test,y_test)
scores_rf=cross_val_score(clf_rf,x_train,y_train)
scores_rf.mean() #94.9%
importances = clf_rf.feature_importances_