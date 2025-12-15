from matplotlib import font_manager, rcParams

font_path = './NanumGothic.ttf'

font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()

rcParams['font.family'] = font_name
rcParams['axes.unicode_minus'] = False

import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from itertools import combinations
from collections import Counter
from wordcloud import WordCloud

st.text('C321011 김다인')
st.title('케이팝 데몬 헌터스 뉴스 분석')

@st.cache_data
def load_processed_data():
    with open('processed_data.pkl', 'rb') as f:
        return pickle.load(f)

data = load_processed_data()

all_nouns = data['all_nouns']
words = data['words']
edge_counts = data['edge_counts']
text_for_cloud = data['text_for_cloud']

results = pd.read_csv('./케데헌.csv')
st.write(f'총 데이터 개수: {len(results)}')
st.dataframe(results.head())

# 단어별 출현 빈도수를 비율로 반환하는 객체를 생성
words_han = WordCloud().generate(text_for_cloud)

st.divider() # 구분선

#seaborn으로 상위 20개 단어의 단어별 출현 빈도 시각화==========================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# 상위 20개 단어 추출
counts = Counter(words)
top20_words = counts.most_common(20)

df_top20 = pd.DataFrame(top20_words, columns=['keyword', 'count'])


#================AI 코드 참조(그래프 생성)==================
# 막대 그래프 그리기
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=df_top20, 
            x='count', 
            y='keyword', 
            palette='viridis',
            hue='keyword')
ax.set_title('상위 20개 단어 출현 빈도')
ax.set_xlabel('출현 빈도')
ax.set_ylabel('단어')

st.subheader('상위 20개 단어 출현 빈도')

st.pyplot(fig)
#===========================================================


#  분석 결과 서술=======================================================================
st.subheader('분석 결과')
st.markdown(
    '''
    뉴스 텍스트 기반 키워드 중요도 분석 결과 브랜드와의 협업이 이슈가 된 것으로 보인다.  
    이후 굿즈/이벤트 등 참여형 이벤트를 통해 점차 확장되었다.  

    나아가 해외 시상식과 해외 매체 언급, 한국 국위선양이 더해지며  
    팬덤은 단순 소비 집단을 넘어 문화적 정체성을 공유하는 커뮤니티로 강화되었음을 확인할 수 있다.
    '''
)

st.divider() # 구분선


# 워드클라우드 시각화 =======================================================
from matplotlib import font_manager

# 한글 폰트 경로 직접 설정
han_font_path = './NanumGothic.ttf'

wc = WordCloud(
    font_path = './NanumGothic.ttf',
    width=1000,
    height=1000,
    background_color='white',
    max_words=100).generate(text_for_cloud)

#================AI 코드 참조(그래프 생성)==================
st.subheader('워드클라우드 시각화')
fig2, ax2 = plt.subplots(figsize=(8,8))
ax2.imshow(wc)
ax2.axis('off')
st.pyplot(fig2)
#===========================================================

# 분석 결과 서술=======================================================
st.subheader('분석 결과')
st.markdown(
    '''
    워드클라우드에서 가장 크게 나타난 키워드는 '농심','신라면','캐릭터','패키지','협업','글로벌','이벤트','굿즈' 등이다.
    이를 보았을 때, 애니메이션 자체보다 외부 요소가 중심을 차지함을 알 수 있다. 특히 농심과 신라면이 많이 등장하는 것을 보았을 때 이른바 '케데헌'은 콘텐츠가 생활 속으로 확장되어, 팬덤 형성의 출발점이 작품 감상이 아닌 일상 속에서의 경험임을 확인할 수 있다.    
    '''
)

st.divider() # 구분선

#  네트워크 그래프 생성==================================================
# edge 리스트 생성
edge_list = []

# 각 문서의 명사 목록에서 2-튜플 조합 생성
for nouns in all_nouns:
    if len(nouns) > 1:  # 단어가 2개 이상인 경우에만 처리 | 단어가 1개 뿐이면 만들 수 있는 쌍이 없음 => 엣지 생성 대상에서 제외
        # 사전식으로 정렬한 후 조합을 생성하여 edge_list에 추가
        edge_list.extend(combinations(sorted(nouns), 2)) # 단어 2개씩 묶기 nC2, 정렬 안 하면 순서 다른 같은 쌍을 다른 엣지로 인식!


# 지정된 최소 빈도 이상의 엣지만 필터링
min_count = 20  # 20번 이상 등장한 관계만 유지하고 나머지는 제거 | 엣지 폭발 방지
filtered_edges = {edge: weight for edge, weight in edge_counts.items() if weight >= min_count}

# 그래프 객체 생성
G = nx.Graph()

# 가중치가 포함된 엣지 리스트 생성
weighted_edges = [
    (node1, node2, weight)  # ex: ('사업', '정비'), 125) => 앞의 두개 노드, 마지막 값 가중치
    for (node1, node2), weight in filtered_edges.items()
]

# 그래프 G에 엣지와 가중치 추가  |  ex: G.edges['대출', '정책']['weight'] == 32
G.add_weighted_edges_from(weighted_edges)

# 레이아웃 생성
pos_spring = nx.spring_layout(
    G,          # 그래프 객체
    k = 1,    # 노드 간의 거리
    iterations=80, # 시뮬레이션 반복 횟수 | 너무 크면 계산 느림
    seed=42   # 시드 고정 | 같은 데이터면 그림이 같아야 함!
)

# 노드 크기 설정 (차수에 비례 | 연결된 엣지 수에 비례)
node_sizes = [G.degree(node) * 200 for node in G.nodes()]

# 엣지 두께  (동시출현 빈도에 비례 | 자주 함께 등장한 키워드 쌍일수록 더 굵게)
edge_widths = [G[u][v]['weight'] * 0.05 for u, v in G.edges()]

#================AI 코드 참조(그래프 생성)==================
#시각화
fig, ax = plt.subplots(figsize=(15,15))

nx.draw_networkx(
    G,
    pos_spring,
    with_labels=True,
    node_size=node_sizes,
    width=edge_widths,
    font_family=plt.rcParams['font.family'],
    font_size=12,
    node_color='skyblue',
    edge_color='gray',
    alpha=0.8,
    ax=ax
)
st.subheader('네트워크 시각화')
st.pyplot(fig)
#========================================================

st.divider() # 구분선

# 상위 50개 엣지 네트워크 시각화====================================
st.subheader('상위 50개 엣지 중 선택하여 네트워크 시각화')

# ============================ AI 코드 참조 ==========================
# 상위 50개 엣지 추출
top30_edges = dict(edge_counts.most_common(50))
# selectbox용 키워드 목록 생성
top30_words = sorted(
    list(set([word for edge in top30_edges.keys() for word in edge]))
)

selected_word = st.selectbox(
    '분석하고 싶은 키워드를 선택하세요',
    sorted(top30_words)
)

# 선택된 단어가 포함된것만 필터링
filtered_top30_edges = {
    edge: weight
    for edge, weight in top30_edges.items()
    if selected_word in edge
}
# ==================================================================

# 그래프 객체 생성
G_selected = nx.Graph()
# 가중치가 포함된 엣지 리스트 생성
weighted_edges_selected = [
    (node1, node2, weight)  # ex: ('사업', '정비'), 125) => 앞의 두개 노드, 마지막 값 가중치
    for (node1, node2), weight in filtered_top30_edges.items()
]
# 그래프 G에 엣지와 가중치 추가  |  ex: G.edges['대출', '정책']['weight'] == 32
G_selected.add_weighted_edges_from(weighted_edges_selected)
# 레이아웃 생성
pos_spring_selected = nx.spring_layout(
    G_selected,          # 그래프 객체
    k = 1,    # 노드 간의 거리
    iterations=80, # 시뮬레이션 반복 횟수 | 너무 크면 계산 느림
    seed=42   # 시드 고정 | 같은 데이터면 그림이 같아야 함!
)
# 노드 크기 설정 (차수에 비례 | 연결된 엣지 수에 비례)
node_sizes_selected = [G_selected.degree(node) * 500 for node in G_selected.nodes()]
# 엣지 두께  (동시출현 빈도에 비례 | 자주 함께 등장한 키워드 쌍일수록 더 굵게)
edge_widths_selected = [G_selected[u][v]['weight'] * 0.1 for u, v in G_selected.edges()]
#시각화
fig_selected, ax_selected = plt.subplots(figsize=(10,10))
nx.draw_networkx(
    G_selected,
    pos_spring_selected,
    with_labels=True,
    node_size=node_sizes_selected,
    width=edge_widths_selected,
    font_family=plt.rcParams['font.family'],
    font_size=12,
    node_color='red',
    edge_color='gray',
    alpha=0.8,
    ax=ax_selected
)
st.pyplot(fig_selected)

####
st.markdown(
    "**연결관계가 많은 단어를 집중적으로 볼 수 있도록 선택하여 확인할 수 있는 네트워크 시각화를 해 보았습니다.**"
)

# 분석 결과 서술=======================================================
st.subheader('분석 결과')
st.markdown(
    '''
    네트워크 시각화에서 확인할 수 있는 특이점으로는 ‘매기’와 ‘감독’이라는 키워드가 하나의 강한 연결 축을 형성하고 있다는 점이다.

해당 연결의 배경을 파악하기 위해 추가적으로 조사한 결과, ‘매기’는 작품의 감독 이름이며, 해당 감독은 한국계 캐나다인인 것으로 확인되었다.
소니 픽처스가 제작한 한국을 배경으로 한 작품의 연출을 한국계 감독이 맡았다는 것은 국내 관객들에게 문화적 친밀감과 자긍심을 자극하는 요소로 작용했을 가능성이 매우 높다.

이는 작품에 대한 관심이 단순히 콘텐츠 자체에 국한되지 않고 제작 배경과 창작 주체의 정체성까지 확장되어 소비되고 있음을 의미한다.
결과적으로 이러한 요소가 한국 관객 및 팬덤의 몰입도를 높이고 작품에 대한 긍정적인 여론 형성에 기여했을 것으로 보인다.   
    '''
)
st.divider() # 구분선

# plotly piechart로 팬덤 구성 비율 시각화=======================================================
# ================== AI 코드 참조(그래프 생성) ==================
import plotly.express as px
from collections import Counter

# 키워드 카테고리 정의
category_map = {
    '농심': '브랜드/협업', '신라면': '브랜드/협업', '패키지': '브랜드/협업', '협업': '브랜드/협업',
    '캐릭터': '콘텐츠', '세계관': '콘텐츠', '작화': '콘텐츠', '감독': '콘텐츠',
    '굿즈': '소비/마케팅', '이벤트': '소비/마케팅', '멤버십': '소비/마케팅',
    '글로벌': '글로벌/문화', '해외': '글로벌/문화', '시상식': '글로벌/문화', '골든': '글로벌/문화'
}

# 전체 단어 빈도 계산
word_counts = Counter(words)

df_words = pd.DataFrame(
    word_counts.items(),
    columns=['keyword', 'count']
)

df_words['category'] = df_words['keyword'].map(category_map).fillna('기타')

# 카테고리별로 집계
df_category = (
    df_words
    .groupby('category')['count']
    .sum()
    .reset_index()
    .sort_values(by='count', ascending=False)
)

df_category = df_category[df_category['category'] != '기타']

# 파이 차트 그리기
st.subheader('카테고리별 키워드 언급 비율')

fig = px.pie(
    df_category,
    values='count',
    names='category',
    hole=0.4,
    color='category',
    color_discrete_map={
        '브랜드/협업': '#FF9999',
        '콘텐츠': '#66B2FF',
        '소비/마케팅': '#99FF99',
        '글로벌/문화': '#FFCC00',
    }
)

fig.update_traces(
    textposition='inside',
    textinfo='percent+label'
)

st.plotly_chart(fig, use_container_width=True)
# =======================================================

# 분석 결과 서술=======================================================
st.subheader('분석 결과')
st.markdown(
    '''
전체 비율 중약 절반이 브랜드/협업 영역에 집중되어 있어 케이팝 데몬 헌터스에 대한 관심이 작품 자체보다 협업을 통해 크게 확산되었음을 알 수 있다.
글로벌/문화와 콘텐츠 영역은 비슷한 비중을 차지하며, 해외 반응과 문화적 상징성, 캐릭터/세계관 등 서사가 함께 소비되고 있음을 보여준다.
반면 소비/마케팅 영역의 비중은 상대적으로 낮아 굿즈나 이벤트는 보조적 요소로 작용한 것으로 해석된다.
따라서 팬덤은 콘텐츠 감상에서 출발해 브랜드 협업과 글로벌 문화로 형성되었다고 말할 수 있다.
이는 일상에서의 소비와 문화적인 자긍심이 결합되어 팬덤이 형성되었음을 알려준다.  
    '''
)
st.divider() # 구분선

# altair 히트맵 시각화=======================================================
# ================== AI 코드 참조(그래프 생성) ==================
import streamlit as st
import altair as alt
import pandas as pd

# 상위 30개 엣지 사용
top_edges = edge_counts.most_common(30)

df_edges = pd.DataFrame(
    top_edges,
    columns=['pair', 'count']
)

# 단어 쌍 분리
df_edges[['Variable1', 'Variable2']] = pd.DataFrame(
    df_edges['pair'].tolist(),
    index=df_edges.index
)
df_edges = df_edges[['Variable1', 'Variable2', 'count']]

# 히트맵
heatmap = alt.Chart(df_edges).mark_rect().encode(
    x=alt.X('Variable1:N', title=None),
    y=alt.Y('Variable2:N', title=None),
    color=alt.Color(
        'count:Q',
        scale=alt.Scale(scheme='blues'),
        legend=alt.Legend(title='동시출현 빈도')
    ),
    tooltip=['Variable1', 'Variable2', 'count']
).properties(
    width=500,
    height=500
)

st.subheader('키워드 간 동시출현 히트맵')
st.altair_chart(heatmap, use_container_width=True)
# =======================================================
# 분석 결과 서술=======================================================

st.subheader('분석 결과')
st.markdown(
    '''
히트맵을 보면 패키지–협업 축이 가장 짙게 나타나며 키워드 동시출현의 중심을형성한다. 이는 케이팝 데몬 헌터스가 콘텐츠 자체보다 패키지 상품을 중심으로 확산되었음을 의미한다.
또한 '신라면–농심–패키지’ 구간에서도 비교적 높은 빈도가 확인되어, 특정 브랜드 제품이 캐릭터/세계관과 결합된 형태로 반복 언급되고 있음을 알 수 있다.
'매기-감독' 또한 브랜드 키워드처럼 강한 연결 고리를 형성하고 있는데, 이는 앞서 언급한 바와 같이 한국계 감독의 참여가 작품에 대한 관심을 증폭시키는 요인이었기 때문으로 보인다.
    '''
)
st.divider() # 구분선


st.subheader('종합 결론')
st.markdown(
    '''
앞선 분석을 통해 케이팝 데몬 헌터스는 작품의 서사나 완성도 자체보다 브랜드 협업과 상품화 전략을 중심으로 형성되었음을 확인하였다.
상위 키워드 분석과 네트워크, 히트맵 시각화 결과, 농심/신라면/패키지/협업과 같은 키워드가 중심 축을 이루며 콘텐츠가 일상적 소비 경험과 결합되어 확산되는 양상을 보였다.
또한 글로벌/시상식/해외 키워드의 비중은 작품이 단순한 애니메이션을 넘어 한국 문화와 정체성을 매개로 한 글로벌 문화 콘텐츠로 인식되고 있음을 보여준다.
특히 감독과 관련된 키워드의 강한 연결은 제작 주체의 정체성 또한 작품 소비의 중요한 요소로 작용했음을 시사한다.
종합적으로 케이팝 데몬 헌터스는 콘텐츠 감상 → 브랜드 협업 → 글로벌 문화 담론으로 확장되며, 소비·문화·정체성이 결합된 팬덤 형성 구조를 만들어낸 사례로 해석할 수 있다.
    '''
)

















