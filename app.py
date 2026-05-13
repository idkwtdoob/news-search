import streamlit as st
import pandas as pd
import json
import re
from google import genai
from google.genai import types
from supabase import create_client, Client

# 1. 환경 설정 및 시크릿 불러오기
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# 클라이언트 초기화
client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="AI 뉴스 커넥터", layout="wide")

# --- 헬퍼 함수: JSON 파싱 ---
def extract_json(text):
    try:
        # ```json ... ``` 형태나 그냥 JSON 문자열에서 배열만 추출
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except:
        return None

# --- 메인 화면 ---
st.title("🚀 AI 최신 뉴스 검색 & 자동 저장기")

tab1, tab2, tab3 = st.tabs(["🔍 검색하기", "💾 저장된 뉴스 보기", "📊 통계 분석"])

# --- Tab 1: 검색 및 저장 ---
with tab1:
    keyword = st.text_input("검색하고 싶은 뉴스 키워드를 입력하세요", placeholder="예: 생성형 AI 트렌드")
    search_btn = st.button("뉴스 검색 및 자동 저장")

    if search_btn and keyword:
        with st.spinner("Gemini가 최신 뉴스를 검색하고 분석 중입니다..."):
            try:
                # 1. Gemini Search 호출 (JSON 모드 미사용, 도구만 사용)
                prompt = f"'{keyword}'에 대한 가장 최신 뉴스 딱 2건만 검색해. 제목, 출처, 날짜, 원본 URL, 요약을 포함한 JSON 배열로 응답하고 절대 URL을 지어내지 마."
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash", # 현재 안정적인 최신 모델 사용
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearchRetrieval())],
                        temperature=0.0
                    ),
                    contents=prompt
                )

                # 2. 결과 텍스트에서 JSON 추출
                news_items = extract_json(response.text)
                
                # 3. [중요] URL 환각 방지 로직 (Grounding Metadata 활용)
                grounding_metadata = response.candidates[0].grounding_metadata
                if grounding_metadata and grounding_metadata.grounding_chunks:
                    chunks = grounding_metadata.grounding_chunks
                    
                    for item in news_items:
                        for chunk in chunks:
                            if chunk.web:
                                # 생성된 제목과 검색 결과 제목이 유사하면 실제 URL로 교체
                                if item['title'][:10] in chunk.web.title or chunk.web.title[:10] in item['title']:
                                    real_url = chunk.web.uri
                                    # 리다이렉트 링크가 아니고 http로 시작하는 경우만 허용
                                    if "grounding-api-redirect" not in real_url and real_url.startswith("http"):
                                        item['url'] = real_url
                
                # 4. 화면 표시 및 DB 저장
                success_count = 0
                skip_count = 0

                for item in news_items:
                    # 화면 출력
                    with st.container():
                        st.markdown(f"### [{item['title']}]({item['url']})")
                        st.caption(f"출처: {item.get('source', '알 수 없음')} | 날짜: {item.get('news_date', '-')}")
                        st.write(item.get('summary', '요약 정보 없음'))
                        st.divider()

                    # DB 저장
                    try:
                        data = {
                            "keyword": keyword,
                            "title": item['title'],
                            "source": item.get('source', ''),
                            "news_date": item.get('news_date', ''),
                            "url": item['url'],
                            "summary": item.get('summary', '')
                        }
                        supabase.table("news_history").insert(data).execute()
                        success_count += 1
                    except Exception as e:
                        if "23505" in str(e): # UNIQUE 제약 조건 위반 에러 코드
                            skip_count += 1
                        else:
                            st.error(f"저장 중 오류: {e}")

                st.toast(f"✅ 완료! 새 저장: {success_count}건 / 중복 생략: {skip_count}건")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# --- Tab 2: 저장된 뉴스 보기 ---
with tab2:
    st.subheader("저장된 뉴스 아카이브")
    
    # 데이터 불러오기
    res = supabase.table("news_history").select("*").order("created_at", desc=True).execute()
    if res.data:
        df = pd.DataFrame(res.data)
        
        # 필터링 UI
        search_filter = st.text_input("제목 또는 키워드 내에서 검색", "")
        filtered_df = df[df['title'].str.contains(search_filter) | df['keyword'].str.contains(search_filter)]
        
        # 데이터프레임 표시
        st.dataframe(filtered_df[['keyword', 'title', 'source', 'news_date', 'url', 'created_at']], use_container_width=True)
        
        # CSV 다운로드
        csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("CSV 데이터 다운로드", csv, "news_export.csv", "text/csv")
    else:
        st.info("아직 저장된 뉴스가 없습니다.")

# --- Tab 3: 통계 분석 ---
with tab3:
    st.subheader("데이터 분석 대시보드")
    
    if res.data:
        df = pd.DataFrame(res.data)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("📌 키워드별 누적 검색 건수")
            keyword_counts = df['keyword'].value_counts()
            st.bar_chart(keyword_counts)
            
        with col2:
            st.write("📅 일자별 저장 추이")
            df['date_only'] = pd.to_datetime(df['created_at']).dt.date
            date_counts = df.groupby('date_only').size()
            st.line_chart(date_counts)
    else:
        st.info("데이터가 충분하지 않습니다.")