import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# 데이터 불러오기
filename = "./data/data.csv"
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(filename, names=column_names)

# 데이터 확인
print(data.shape)
print(data.describe())
print(data.groupby('class').size())  # 'class'에 따른 그룹 크기 확인

# scatter_matrix 그래프 저장
scatter_matrix(data, figsize=(10, 10))
plt.savefig("scatter_matrix.png")

# 독립변수와 x와 종속변수 y로 분할
X = data.iloc[:, 0:4]
Y = data.iloc[:, 4]

# 모델 정의
model = DecisionTreeClassifier()

# 모델 학습
model.fit(X, Y)

# 예측
y_pred = model.predict(X)
print(y_pred)

# K-fold(10개의 폴드 지정) 및 cross validation(평가 지표 accuracy)
kfold = KFold(n_splits=10, random_state=10, shuffle=True)
results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
print(results.mean())
