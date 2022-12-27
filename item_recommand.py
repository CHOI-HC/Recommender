#%%
import pandas as pd
import numpy as np
from numpy import dot

from dataprep.eda import plot, create_report

from ast import literal_eval

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error 
import warnings; warnings.filterwarnings('ignore')
# %%
movies = pd.read_csv(r'data/grouplens_movies.csv')
ratings = pd.read_csv(r'data/grouples_ratings.csv')
print(f"movies shape: {movies.shape}")
print(f"movies shape: {ratings.shape}")

mvCol = list(movies.columns)
rtCol = list(ratings.columns)
comCol = list(set(mvCol) & set(rtCol))
# print(comcol)

df = pd.merge(ratings, movies, how='inner', on=comCol)
print(f"rating and movie merge shape: {df.shape}")

pivotDf = pd.pivot_table(data=df, index='userId', columns='title', values='rating')
print(f'pivotdf shape: {pivotDf.shape} > 최종 dataframe')

pivotDf = pivotDf.fillna(0) #최종 dataframe
pivotDf_trans = pivotDf.transpose()
print(f'transpose pivotDf shape: {pivotDf_trans.shape}')

#코샤인 유사도 확인
itemSim = cosine_similarity(pivotDf_trans, pivotDf_trans)
#유사도 dataframe
simDf = pd.DataFrame(itemSim, index=pivotDf_trans.index, columns=pivotDf_trans.index)
print(f'similarity array shape: {simDf.shape} > 유사도 dataframe')

#Inception과 유사도가 가장 비슷한 10개 영화
simDf['Inception (2010)'].sort_values(ascending=False)[:11]
print('\n\t<top10 similarity of Inception>\n', simDf["Inception (2010)"].sort_values(ascending=False))
print('순서: 최종df(완료) > 유사도df(완료) > 최종df * 유사도df: 행렬곱 > 새로운 df')

fDf = dot(pivotDf, simDf)
dotDf = pd.DataFrame(fDf, index=pivotDf.index, columns=pivotDf.columns)
print(f"final dataframe shape: {dotDf.shape}")
rpred = dotDf / np.array([abs(simDf.values).sum(axis=1)]) #예측평점
rpredDf = pd.DataFrame(rpred, index=pivotDf.index, columns=pivotDf.columns)
print(f"rpredDf shape: {rpredDf.shape}")
rpredDf.head()

#최종 dataframe, 예측평점 dataframe의 mse 계산
nzAct = pivotDf.values[np.nonzero(pivotDf.values)]
nzPred = rpredDf.values[np.nonzero(pivotDf.values)]
mse = mean_squared_error(nzAct, nzPred)
print(f"pivotDf and predDf mean squared error: {mse}")

rpred2 = np.zeros(pivotDf.shape)
print(f"rpred2 shape: {rpred2.shape}")
for col in range(pivotDf.shape[1]):
    top20sim = np.argsort(simDf.values[:,col])[:-21:-1]
    for row in range(pivotDf.shape[0]):
        rpred2[row, col] = simDf.values[col,:][top20sim].dot(pivotDf.values[row,:][top20sim])
        rpred2[row, col] /= np.abs(simDf.values[col,:][top20sim]).sum()
rpred2Df = pd.DataFrame(rpred2, index=pivotDf.index, columns=pivotDf.columns)
print(f"rpred2Df shape: {rpred2Df.shape}")

#%%
#mse 계산
nzAct2 = pivotDf.values[np.nonzero(pivotDf.values)]
nzPred2 = rpred2Df.values[np.nonzero(pivotDf.values)]
mse2 = mean_squared_error(nzAct2, nzPred2)
print(f"pivotDf and pred2Df mean squared error: {mse2}")
# %%
user9 = pivotDf.loc[9, :] #user9가 어떤 영화를 좋아하는지 확인
print(user9[user9>0].sort_values(ascending=False)[:10])
#본 영화 제외
idSeries = pivotDf.loc[9,:] #user9
alrSeen = idSeries[idSeries>0].index.tolist() #이미 본 영화 리스트
allMovies = list(pivotDf)
unSeen = [i for i in alrSeen if i not in allMovies]

#user9에게 추천
print('\t<아이템 기반 협업필터링: 영화추천(user9)>\n', rpred2Df.loc[9, unSeen].sort_values(ascending=False)[:10])

itRecomDf = pd.DataFrame(rpred2Df.loc[9, unSeen].sort_values(ascending=False)[:10])
itRecomDf.rename(columns={9:'pred'}, inplace=True)
itRecomDf
