import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────
st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

LAST_YEAR = 2025  # 수업의 기준 기간: 2025년까지만 사용합니다.
MIN_DAYS = 300    # 관측일이 300일 미만인 해는 연평균이 믿기 어려워서 뺍니다.


@st.cache_data(ttl=3600)  # 1시간 동안 같은 데이터를 다시 안 불러오고 기억해 둡니다.
def load_yearly_temperature():
    """
    서울 일별 기온 데이터를 불러와서 '연도별 평균기온' 표를 만들어 돌려줍니다.
    1) 날짜를 진짜 날짜로, 평균기온을 숫자로 바꿉니다. (바꿀 수 없는 값은 빈칸 처리)
    2) 2025년 이후 데이터를 뺍니다.
    3) 연도별로 평균기온의 평균(연평균기온)과 관측일수를 구합니다.
    4) 관측일수가 300일 미만인 해를 뺍니다.
    """
    raw = pd.read_csv(DATA_URL, encoding="utf-8")
    raw.columns = raw.columns.str.strip()  # 열 이름 앞뒤 공백 제거

    raw["날짜"] = pd.to_datetime(raw["날짜"].astype(str).str.strip(), errors="coerce")
    raw["평균기온"] = pd.to_numeric(raw["평균기온"], errors="coerce")

    # 날짜나 평균기온이 비어 있는 행은 '관측되지 않은 날'로 보고 뺍니다.
    raw = raw.dropna(subset=["날짜", "평균기온"])

    # 기준 기간(2025년까지)만 남깁니다.
    raw = raw[raw["날짜"].dt.year <= LAST_YEAR]

    # 같은 날짜가 여러 줄 있을 때를 대비해서 날짜별로 한 번 평균을 냅니다.
    daily = raw.groupby("날짜", as_index=False)["평균기온"].mean()
    daily["연도"] = daily["날짜"].dt.year

    # 연도별 연평균기온과 관측일수를 구합니다.
    yearly = (
        daily.groupby("연도")
        .agg(연평균기온=("평균기온", "mean"), 관측일수=("평균기온", "count"))
        .reset_index()
    )

    # 관측일이 300일이 안 되는 해는 뺍니다.
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)

    return yearly


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온에 직선(회귀 직선)을 맞춰서, 원하는 해의 기온을 예상해 봐요.")

# ──────────────────────────────────────────────
# 데이터 불러오기 (실패하면 안내문을 보여주고 멈춥니다)
# ──────────────────────────────────────────────
try:
    yearly = load_yearly_temperature()
except Exception as e:
    st.error(
        "⚠️ 데이터를 불러오지 못했어요.\n\n"
        "인터넷 연결과 데이터 주소, 열 이름(날짜·평균기온)을 확인해 주세요.\n\n"
        f"(오류 내용: {e})"
    )
    st.stop()

if len(yearly) < 2:
    st.error("⚠️ 직선을 만들 수 있는 해가 2개 미만이에요. 데이터와 기준(2025년까지, 관측일 300일 이상)을 확인해 주세요.")
    st.stop()

# ──────────────────────────────────────────────
# 회귀 직선과 상관계수 계산
# ──────────────────────────────────────────────
years = yearly["연도"].to_numpy()
temps = yearly["연평균기온"].to_numpy()

# np.polyfit(x, y, 1) = 1차식(직선) y = 기울기 * x + 절편 을 찾아줍니다.
slope, intercept = np.polyfit(years, temps, 1)

# 상관계수(r): 1이나 -1에 가까울수록 직선 관계가 뚜렷합니다.
corr = np.corrcoef(years, temps)[0, 1]

start_year = int(years.min())
end_year = int(years.max())
n_years = len(yearly)

# ──────────────────────────────────────────────
# 연도 슬라이더와 예상 기온
# ──────────────────────────────────────────────
st.header("1. 연도를 골라 예상 기온 보기")

selected_year = st.slider("연도를 고르세요", min_value=1900, max_value=2100, value=2030)

predicted = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:1.2rem;">{selected_year}년의 예상 연평균기온</div>
        <div style="font-size:4rem; font-weight:700;">{predicted:.1f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_year < start_year or selected_year > end_year:
    st.caption(
        f"⚠️ {selected_year}년은 직선을 만든 기간({start_year}~{end_year}년) 밖이에요. "
        "기간에서 멀어질수록 예상값은 덜 믿을 수 있어요."
    )

st.divider()

# ──────────────────────────────────────────────
# 산점도 + 회귀 직선
# ──────────────────────────────────────────────
st.header("2. 연평균기온 산점도와 회귀 직선")

# 직선은 데이터 기간과 선택한 연도를 모두 덮도록 그립니다.
line_x = np.array([min(start_year, selected_year), max(end_year, selected_year)])
line_y = slope * line_x + intercept

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=yearly["연도"],
    y=yearly["연평균기온"],
    mode="markers",
    name="연평균기온",
    marker=dict(size=8, color="#4C78A8"),
    hovertemplate="연도: %{x}년<br>연평균기온: %{y:.2f}℃<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode="lines",
    name="회귀 직선",
    line=dict(color="red", width=2),
    hovertemplate="연도: %{x}년<br>직선 값: %{y:.2f}℃<extra></extra>",
))

# 슬라이더로 고른 연도의 예상 기온을 별 모양으로 표시합니다.
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted],
    mode="markers",
    name=f"{selected_year}년 예상",
    marker=dict(size=16, color="orange", symbol="star", line=dict(color="black", width=1)),
    hovertemplate="연도: %{x}년<br>예상 기온: %{y:.2f}℃<extra></extra>",
))

fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온(℃)",
)

st.plotly_chart(fig, use_container_width=True, key="fig_temp_regression")

# ──────────────────────────────────────────────
# 상관계수와 직선을 만든 데이터 정보
# ──────────────────────────────────────────────
st.header("3. 직선을 만든 데이터")

col1, col2, col3, col4 = st.columns(4)
col1.metric("상관계수 (r)", f"{corr:.3f}")
col2.metric("직선을 만든 해의 개수", f"{n_years}개")
col3.metric("시작 연도", f"{start_year}년")
col4.metric("끝 연도", f"{end_year}년")

st.write(f"회귀 직선: **연평균기온 = {slope:.4f} × 연도 + ({intercept:.2f})**  (10년에 약 {slope * 10:+.2f}℃ 변화)")
st.caption(
    f"기준: {LAST_YEAR}년까지의 데이터 중 관측일이 {MIN_DAYS}일 이상인 해만 사용했어요."
)
