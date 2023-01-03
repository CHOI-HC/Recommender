#%%
import pandas as pd
import numpy as np
from numpy import dot

from ast import literal_eval

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error 
import warnings; warnings.filterwarnings('ignore')
# %%
movies = pd.read_csv(r'data/grouplens_movies.csv')
ratings = pd.read_csv(r'data/grouples_ratings.csv')
print(f"movies shape: {movies.shape}")
print(f"movies shape: {ratings.shape}")

mvCol = list(movies.columns)  # movies dataframe 컬럼들
rtCol = list(ratings.columns)  # ratings dataframe 컬럼들
comCol = list(set(mvCol) & set(rtCol))  # movies, ratings 공통 컬럼

#%%
df = pd.merge(ratings, movies, how='inner', on=comCol)  # 공통컬럼(comCOl)을 기준으로 inner merge
print(f"rating and movie merge shape: {df.shape}")


#%%
# usderId가 인덱스, 영화제목이 컬럼명, 평점이 값으로 들어간 pivotDf 생성
pivotDf = pd.pivot_table(data=df, index='userId', columns='title', values='rating') 
print(f'pivotdf shape: {pivotDf.shape} > 최종 dataframe')
pivotDf.head(30)

#%%
# NaN 0으로 채우기
pivotDf = pivotDf.fillna(0) #최종 dataframe

pivotDf_trans = pivotDf.transpose() # 영화제목을 인덱스로, usdrId 컬럼으로
print(f'transpose pivotDf shape: {pivotDf_trans.shape}')
pivotDf_trans.head(10)

#%%
# 코사인 유사도 확인(영화들 간의 유사도)
itemSim = cosine_similarity(pivotDf_trans, pivotDf_trans)

# 유사도 dataframe
simDf = pd.DataFrame(itemSim, index=pivotDf_trans.index, columns=pivotDf_trans.index)
print(f'similarity array shape: {simDf.shape} > 유사도 dataframe')
simDf.head(10)

#%%
# (예시)Inception과 유사도가 가장 비슷한 10개 영화
simDf['Inception (2010)'].sort_values(ascending=False)[:11]
print('\n\t<top10 similarity of Inception>\n', simDf["Inception (2010)"].sort_values(ascending=False))
print('순서: 최종df(완료) > 유사도df(완료) > 최종df * 유사도df: 행렬곱 > 새로운 df')

#%%
# pivotDf, simDf 간의 행렬곱
final_df = dot(pivotDf, simDf)
final_df
#%%
# 행렬곱된 fianl_df의 index를 usdrid, columns를 titles로 설정하여 dotDf에 할당
dotDf = pd.DataFrame(final_df, index=pivotDf.index, columns=pivotDf.columns)
print(f"final dataframe shape: {dotDf.shape}")
dotDf.head(10)

#%%
# 예측평점
rpred = dotDf / np.array([abs(simDf.values).sum(axis=1)])
rpredDf = pd.DataFrame(rpred, index=pivotDf.index, columns=pivotDf.columns)
print(f"rpredDf shape: {rpredDf.shape}")

#%%
## 최종 dataframe, 예측평점 dataframe의 mse 계산
# np.nonzero(): 0이 아닌 값들의 인덱스 반환
nzAct = pivotDf.values[np.nonzero(pivotDf.values)]  # 실제로 평점을 매긴 영화들의 평점들
nzPred = rpredDf.values[np.nonzero(pivotDf.values)]  #  실제로 평점을 매긴 영화들에 대한 예측 평점들 
mse = mean_squared_error(nzAct, nzPred)
print(f"pivotDf and predDf mean squared error: {mse}")

#%%
rpred2 = np.zeros(pivotDf.shape)
print(f"rpred2 shape: {rpred2.shape}")
for col in range(pivotDf.shape[1]):  # range(9719)
    top20sim = np.argsort(simDf.values[:,col])[:-21:-1]  # np.argsort(): 작은 값부터 정렬, # 뒤에서 20개(큰 값)
    for row in range(pivotDf.shape[0]):  # range(610)
        ## [0,0] > [1,0], [2,0] ...
        rpred2[row, col] = simDf.values[col,:][top20sim].dot(pivotDf.values[row,:][top20sim])
        rpred2[row, col] /= np.abs(simDf.values[col,:][top20sim]).sum()  # 예측 평점 만들기
rpred2Df = pd.DataFrame(rpred2, index=pivotDf.index, columns=pivotDf.columns)
print(f"rpred2Df shape: {rpred2Df.shape}")

#%%
#mse 계산
nzAct2 = pivotDf.values[np.nonzero(pivotDf.values)]
nzPred2 = rpred2Df.values[np.nonzero(pivotDf.values)]
mse2 = mean_squared_error(nzAct2, nzPred2)
print(f"pivotDf and pred2Df mean squared error: {mse2}")

# %%
user610 = pivotDf.loc[610, :]  # user9가 어떤 영화를 좋아하는지 확인
print(user610[user610>0].sort_values(ascending=False)[:10])

#%%
# 본 영화 제외
idSeries = pivotDf.loc[610,:]  # user 610
alrSeen = idSeries[idSeries>0].index.tolist()  # 이미 본 영화 리스트
allMovies = list(pivotDf)
unSeen = [movie for movie in allMovies if movie not in alrSeen]


#user610에게 추천
print('\t<아이템 기반 협업필터링: 영화추천(user610)>\n', rpred2Df.loc[610, unSeen].sort_values(ascending=False)[:10])

movie_rec_df = pd.DataFrame(rpred2Df.loc[610, unSeen].sort_values(ascending=False)[:10])
movie_rec_df.rename(columns={610:'pred'}, inplace=True)
movie_rec_df
# %%
