import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix


df=pd.read_csv("D:\\downloads\\nitishh\\pandas_project\\titanic eda\\titanic.csv")

df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.drop(columns=["PassengerId","Name","Unnamed: 0","Ticket"],axis=1,inplace=True)

df["deck"]=df["Cabin"].str[0]
df["deck"]=df["deck"].fillna("U")

cols=["Survived","Pclass","Sex","Age","Fare","deck","Embarked","SibSp","Parch"]

new_df=df[cols].copy()
new_df.dropna(inplace=True)


X=new_df.drop(columns=["deck"],axis=1)
y=new_df["deck"]

#preprocessing ,scalling and transforming of training and testing data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


processing_deck=ColumnTransformer(transformers=[
    ("ohe",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),["Pclass","Sex","Embarked"]),
    ("scaler",StandardScaler(),["Age","Fare"])
],remainder="passthrough").set_output(transform="pandas")


X_train=processing_deck.fit_transform(X_train)
X_test=processing_deck.transform(X_test)

#randomforest model for multiclass classification



rf=RandomForestClassifier(n_estimators=300,max_depth=10,class_weight="balanced",random_state=42)
rf.fit(X_train,y_train)
pred=rf.predict(X_test)


# model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

precision_deck=precision_score(y_test,pred,average="weighted")
recall_deck=recall_score(y_test,pred,average="weighted")

f1_deck=2*(precision_deck*recall_deck/(precision_deck+recall_deck))


# prediction of deck/cabins
main=df.copy()
main.drop(columns=["Cabin","deck"],inplace=True,axis=1)

test_data=processing_deck.transform(main)

main["deck"]=rf.predict(test_data)


# creating data for survival data
X1=main.drop(columns=["Survived"],axis=1)
y1=main["Survived"]

X_train1, X_test1, y_train1, y_test1=train_test_split(X1,y1,test_size=0.2,random_state=42)

# since deck and fare are intercorreted colinearity but with deck model recall is better

processing_survival=ColumnTransformer(transformers=[
    ("ohe",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),["Pclass","Sex","Embarked","deck"]),
    ("scaler",StandardScaler(),["Age","Fare"])
],remainder="passthrough").set_output(transform="pandas")

X_train1=processing_survival.fit_transform(X_train1)
X_test1=processing_survival.transform(X_test1)


#model & prediction


lr=LogisticRegression(max_iter=1000)
lr.fit(X_train1,y_train1)
pred1=lr.predict(X_test1)

#evalutaion
pre=precision_score(y_test1,pred1,average="binary")
recall=recall_score(y_test1,pred1,average="binary")

f1=2*(pre*recall/(pre+recall))

cm=confusion_matrix(y_test1,pred1,labels=lr.classes_)

ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lr.classes_).plot()
plt.show()

#cross validation
from sklearn.model_selection import cross_val_score
clv=cross_val_score(lr,X_train1, y_train1,cv=5,scoring="f1_macro")

print(clv)
print(clv.mean())