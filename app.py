import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
from google import genai
from google.genai import types
from supabase import create_client

# ----------------------------------------------------
# 1. 초기 설정 및 시크릿 불러오기
# ----------------------------------------------------
st.set_page_config(page_title="AI 최신 뉴스 수집기", page_icon="📰", layout="wide")

# API 및 DB 연결 (st.secrets 활용)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# 클라이언트 초기화
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("📰 AI 최신 뉴스 검색 & 자동 저장기")
st.markdown("키워드를 검색하면 Gemini가 구글 검색을 통해 가장 최신 뉴스 2건을 요약하고 DB에 자동 저장합니다.")

# ----------------------------------------------------
# 화면 탭 구성
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 검색하기", "💾 저장된 뉴스 보기", "📊 통계 분석"])

# ==========================================
# Tab 1: 검색 및 저장 로직
# ==========================================
with tab1:
    st.subheader("새로운 뉴스 검색")
    
    with st.form("search_form"):
        keyword = st.text_input("검색할 키워드를 입력하세요 (예: 인공지능, 테슬라, 한국경제 등)")
        submitted = st.form_submit_button("검색 및 요약하기 🚀")
        
    if submitted and keyword:
        with st.spinner(f"'{keyword}'에 대한 최신 뉴스를 검색하고 분석 중입니다..."):
            try:
                # [중요] JSON 강제 모드와 구글 검색 도구는 동시 사용 불가하므로, 프롬프트로 JSON 형태를 강제함
                prompt = f"""
                '{keyword}'에 대한 가장 최신 뉴스 딱 2건만 구글에서 검색해 줘.
                검색된 결과를 바탕으로 반드시 아래 JSON 배열 형식으로만 응답해. 백틱(```)이나 추가 설명 없이 JSON만 출력해.[
                    {{
                        "title": "기사 제목",
                        "source": "언론사 이름",
                        "news_date": "기사 발행일 (예: 2023-10-25)",
                        "url": "기사 원본 URL",
                        "summary": "기사 내용 3줄 요약"
                    }}
                ]
                절대 URL을 지어내지(환각) 마.
                """
                
                # Gemini API 호출 (온도 0.0, 구글 검색 도구 활성화)
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        tools=[{"google_search": {}}]
                    )
                )
                
                # JSON 텍스트 추출 (마크다운 백틱이 섞여있을 경우 대비)
                response_text = response.text
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                
                if json_match:
                    news_data = json.loads(json_match.group())
                    
                    # ----------------------------------------------------
                    # [매우 중요] URL 환각 완벽 방지 로직 (Grounding Metadata 활용)
                    # ----------------------------------------------------
                    real_links = {}
                    # response의 grounding_metadata에서 실제 구글 검색이 참조한 정보 추출
                    if hasattr(response, 'candidates') and response.candidates:
                        grounding_metadata = response.candidates[0].grounding_metadata
                        if grounding_metadata and grounding_metadata.grounding_chunks:
                            for chunk in grounding_metadata.grounding_chunks:
                                if hasattr(chunk, 'web') and chunk.web:
                                    # 실제 구글 검색된 기사 제목과 URL 맵핑
                                    real_links[chunk.web.title] = chunk.web.uri
                    
                    # 생성된 JSON 데이터의 URL을 실제 URL로 덮어쓰기 검증
                    for item in news_data:
                        for real_title, real_url in real_links.items():
                            # 제목이 일부라도 일치하면 (LLM이 제목을 줄였을 수 있으므로) 실제 URL 할당
                            if item['title'].lower() in real_title.lower() or real_title.lower() in item['title'].lower():
                                #[수정 완료] 구글 내부 임시 링크(grounding-api-redirect)가 아니고 정상적인 http 링크일 때만 덮어쓰기
                                if real_url.startswith("http") and "grounding-api-redirect" not in real_url:
                                    item['url'] = real_url
                                break
                    # ----------------------------------------------------
                    
                    # 화면에 출력 및 DB 저장 처리
                    saved_count = 0
                    skipped_count = 0
                    
                    st.success("✨ 검색이 완료되었습니다!")
                    
                    for idx, item in enumerate(news_data):
                        # 카드 형태로 화면 출력
                        with st.container():
                            st.markdown(f"### {idx+1}. [{item['title']}]({item['url']})")
                            st.caption(f"출처: {item['source']} | 날짜: {item['news_date']}")
                            st.write(f"**요약:** {item['summary']}")
                            st.divider()
                        
                        # Supabase DB 저장
                        try:
                            db_data = {
                                "keyword": keyword,
                                "title": item['title'],
                                "source": item['source'],
                                "news_date": item['news_date'],
                                "url": item['url'],
                                "summary": item['summary']
                            }
                            supabase.table("news_history").insert(db_data).execute()
                            saved_count += 1
                        except Exception as e:
                            # 23505는 PostgreSQL의 Unique Violation 에러 코드
                            if '23505' in str(e) or 'duplicate key' in str(e).lower():
                                skipped_count += 1
                            else:
                                st.error(f"DB 저장 중 오류 발생: {e}")
                    
                    st.toast(f"✅ 새 뉴스 {saved_count}건 저장완료! (중복 생략됨: {skipped_count}건)", icon="🎉")
                else:
                    st.error("데이터를 파싱하는 데 실패했습니다. 다시 시도해 주세요.")
                    
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

# ==========================================
# Tab 2: 저장된 뉴스 보기
# ==========================================
with tab2:
    st.subheader("💾 DB에 저장된 뉴스 히스토리")
    
    # DB에서 데이터 가져오기 (최신순)
    response = supabase.table("news_history").select("*").order("created_at", desc=True).execute()
    data = response.data
    
    if data:
        df = pd.DataFrame(data)
        
        # 날짜 포맷 정리
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 필터링 기능
        filter_text = st.text_input("🔍 제목 또는 키워드로 검색 (결과 내 검색)")
        if filter_text:
            df = df[df['title'].str.contains(filter_text, case=False, na=False) | 
                    df['keyword'].str.contains(filter_text, case=False, na=False)]
        
        # 데이터프레임 출력
        st.dataframe(
            df[['keyword', 'title', 'source', 'news_date', 'url', 'created_at']],
            use_container_width=True,
            hide_index=True
        )
        
        # CSV 다운로드 버튼
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 현재 데이터 CSV로 다운로드",
            data=csv,
            file_name=f"news_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("아직 저장된 뉴스가 없습니다. 탭 1에서 뉴스를 검색해 보세요!")

# ==========================================
# Tab 3: 통계 대시보드
# ==========================================
with tab3:
    st.subheader("📊 뉴스 수집 통계")
    
    if data: # Tab 2에서 불러온 데이터 재활용
        stat_df = pd.DataFrame(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📌 키워드별 누적 수집 건수")
            keyword_counts = stat_df['keyword'].value_counts()
            st.bar_chart(keyword_counts)
            
        with col2:
            st.markdown("##### 📅 일자별 뉴스 저장 건수")
            # created_at에서 YYYY-MM-DD 만 추출
            stat_df['date_only'] = pd.to_datetime(stat_df['created_at']).dt.strftime('%Y-%m-%d')
            date_counts = stat_df['date_only'].value_counts().sort_index()
            st.line_chart(date_counts)
    else:
        st.info("통계를 표시할 데이터가 부족합니다.")