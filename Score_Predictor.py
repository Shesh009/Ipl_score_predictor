import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def score_predict(bat_team,bowl_team,runs,wickets,overs,last_5_runs,last_5_wickets):
    pred_array=[]
    #batting
    if bat_team=="Kolkata Knight Riders" or bat_team=="kkr":
        pred_array=pred_array+[1,0,0,0,0,0,0,0]
    elif bat_team=="Chennai Super Kings" or bat_team=="csk":
        pred_array=pred_array+[0,1,0,0,0,0,0,0]
    elif bat_team=="Mumbai Indians" or bat_team=="mi":
        pred_array=pred_array+[0,0,0,1,0,0,0,0]
    elif bat_team=="Sunrisers Hyderabad" or bat_team=="srh":
        pred_array=pred_array+[0,0,0,0,1,0,0,0]
    elif bat_team=="Kings XI Punjab" or bat_team=="kxip":
        pred_array=pred_array+[0,0,0,0,0,1,0,0]
    elif bat_team=="Royal Challengers Bangalore" or bat_team=="rcb":
        pred_array=pred_array+[0,0,0,0,0,0,1,0]
    elif bat_team=="Delhi Daredevils" or bat_team=="dd":
        pred_array=pred_array+[0,0,0,0,0,0,0,1]
    elif bat_team=="Rajasthan Royals" or bat_team=="rr":
        pred_array=pred_array+[0,0,1,0,0,0,0,0]
    
    #bowling
    if bowl_team=="Kolkata Knight Riders" or bowl_team=="kkr":
        pred_array=pred_array+[1,0,0,0,0,0,0,0]
    elif bowl_team=="Chennai Super Kings" or bowl_team=="csk":
        pred_array=pred_array+[0,1,0,0,0,0,0,0]
    elif bowl_team=="Mumbai Indians" or bowl_team=="mi":
        pred_array=pred_array+[0,0,0,1,0,0,0,0]
    elif bowl_team=="Sunrisers Hyderabad" or bowl_team=="srh":
        pred_array=pred_array+[0,0,0,0,1,0,0,0]
    elif bowl_team=="Kings XI Punjab" or bowl_team=="kxip":
        pred_array=pred_array+[0,0,0,0,0,1,0,0]
    elif bowl_team=="Royal Challengers Bangalore" or bowl_team=="rcb":
        pred_array=pred_array+[0,0,0,0,0,0,1,0]
    elif bowl_team=="Delhi Daredevils" or bowl_team=="dd":
        pred_array=pred_array+[0,0,0,0,0,0,0,1]
    elif bowl_team=="Rajasthan Royals" or bowl_team=="rr":
        pred_array=pred_array+[0,0,1,0,0,0,0,0]
        
    pred_array=pred_array+[runs,wickets,overs,last_5_runs,last_5_wickets]
    pred_array=np.array([pred_array])
    pred=reg1.predict(pred_array)
    return int(round(pred[0]))

data=pd.read_csv("IPL_5.csv")

data=data.replace("Deccan Chargers","Sunrisers Hyderabad")

teams=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals','Mumbai Indians', 
       'Sunrisers Hyderabad', 'Kings XI Punjab','Royal Challengers Bangalore', 'Delhi Daredevils']

data=data[data["bat_team"].isin(teams)]
data=data[data["bowl_team"].isin(teams)]

data=data[["bat_team","bowl_team","runs","wickets","overs","runs_last_5","wickets_last_5","total"]]

team_dict={'Kolkata Knight Riders':0,'Chennai Super Kings':1,'Rajasthan Royals':2,
       'Mumbai Indians':3, 'Sunrisers Hyderabad':4, 'Kings XI Punjab':5,
       'Royal Challengers Bangalore':6, 'Delhi Daredevils':7}

data=data.replace({"bat_team":team_dict})
data=data.replace({"bowl_team":team_dict})

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0,1])],remainder="passthrough")
data=ct.fit_transform(data)

col=["bat_kkr","bat_csk","bat_rr","bat_mi","bat_srh","bat_kxip","bat_rcb","bat_dd","bowl_kkr","bowl_csk","bowl_rr","bowl_mi","bowl_srh","bowl_kxip","bowl_rcb","bowl_dd","runs","wickets","overs","runs_last_5","wickets_last_5","total"]
df=pd.DataFrame(data,columns=col)

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

reg1=RandomForestRegressor(n_estimators=100,random_state=0)
reg1.fit(x_train,y_train)

# col11,col12,col13=st.columns(3)
# col11.image("ipl_set2.png")
# col12.image("ipl_set3.png")
# col13.image("ipl_set1.png")
st.header("IPL Score Predictor")
bat=st.selectbox("Batting Team",teams,index=0)
bowl=st.selectbox("Bowling Team",teams,index=1)
col1,col2=st.columns(2)
over=col1.number_input("Overs Completed",min_value=0.0,max_value=20.0,step=0.1)
run=col2.number_input("Runs Scored",min_value=0,step=1)
wicket=st.slider("Wickets Lost",min_value=0,max_value=10,step=1)
col3,col4,col5=st.columns(3)
run_5=col3.number_input("Runs scored in last 5 overs",min_value=0,step=1)
wicket_5=col5.number_input("Wickets lost in last 5 overs",min_value=0,step=1)
col6,col7,col8,col9,col10=st.columns(5)
if col8.button("PREDICT THE SCORE"):
    x=score_predict(bat_team=bat,bowl_team=bowl,runs=run,wickets=wicket,overs=over,last_5_runs=run_5,last_5_wickets=wicket_5)
    st.success("\tPredicted score may be in between {0} to {1}".format(x-5,x+5))
