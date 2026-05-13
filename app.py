import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from collections import Counter
from google import genai
from google.genai import types
from supabase import create_client


# ----------------------------------------------------
# 1. 초기 설정 및 시크릿 불러오기
# ----------------------------------------------------
st.set_page_config(
    page_title="AI 최신 뉴스 인사이트 분석기",
    page_icon="📰",
    layout="wide"
)

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]


@st.cache_resource
def get_clients():
    """Streamlit 재실행 때마다 클라이언트를 새로 만들지 않도록 캐싱합니다."""
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return gemini_client, supabase_client


gemini_client, supabase = get_clients()


# ----------------------------------------------------
# 2. 공통 유틸 함수
# ----------------------------------------------------
ANALYSIS_COLUMNS = [
    "one_line_summary",
    "background",
    "why_it_matters",
    "stakeholders",
    "impact_analysis",
    "issue_points",
    "sentiment",
    "importance_score",
    "tags",
    "source_type",
    "reliability_note",
]


def is_real_http_url(url: str) -> bool:
    """Gemini grounding 내부 리다이렉트 링크를 제외한 실제 http/https URL만 허용합니다."""
    if not isinstance(url, str):
        return False

    cleaned_url = url.strip()
    return (
        cleaned_url.startswith("http")
        and "grounding-api-redirect" not in cleaned_url
    )


def normalize_text(text: str) -> str:
    """제목 비교를 위해 특수문자와 공백을 단순화합니다."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def title_similarity(a: str, b: str) -> float:
    """생성된 제목과 grounding 제목의 유사도를 계산합니다."""
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    if a_norm in b_norm or b_norm in a_norm:
        return 1.0

    return SequenceMatcher(None, a_norm, b_norm).ratio()


def parse_json_array(response_text: str):
    """
    Gemini가 JSON만 출력하지 않고 ```json 같은 마크다운을 섞어도
    배열 부분만 찾아 파싱합니다.
    """
    if not response_text:
        raise ValueError("Gemini 응답이 비어 있습니다.")

    # ```json ... ``` 제거
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not json_match:
        raise ValueError("JSON 배열을 찾지 못했습니다.")

    parsed = json.loads(json_match.group())
    if not isinstance(parsed, list):
        raise ValueError("응답이 JSON 배열 형식이 아닙니다.")

    return parsed


def extract_grounding_links(gemini_response):
    """Gemini Google Search grounding metadata에서 실제 참조 제목과 URL을 가져옵니다."""
    links = []

    try:
        if not getattr(gemini_response, "candidates", None):
            return links

        grounding_metadata = gemini_response.candidates[0].grounding_metadata
        if not grounding_metadata or not grounding_metadata.grounding_chunks:
            return links

        for chunk in grounding_metadata.grounding_chunks:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "title", None) and getattr(web, "uri", None):
                links.append({
                    "title": web.title,
                    "url": web.uri
                })
    except Exception:
        return links

    return links


def overwrite_urls_with_grounding(news_data, grounding_links):
    """
    URL 환각 방지 핵심 로직:
    생성된 JSON의 title과 Gemini가 실제 검색에서 참조한 grounding title을 비교해서,
    일치도가 높을 때만 실제 grounding URL로 덮어씁니다.
    """
    for item in news_data:
        item_title = item.get("title", "")
        best_match = None
        best_score = 0.0

        for link in grounding_links:
            score = title_similarity(item_title, link["title"])
            if score > best_score:
                best_score = score
                best_match = link

        if best_match and best_score >= 0.45 and is_real_http_url(best_match["url"]):
            item["url"] = best_match["url"]
            item["reliability_note"] = (
                f"Google Search grounding metadata의 실제 URL로 교체됨 "
                f"(제목 유사도: {best_score:.2f})"
            )
        else:
            original_url = item.get("url", "")
            if is_real_http_url(original_url):
                item["reliability_note"] = (
                    "URL 형식은 유효하지만 grounding 제목 매칭은 실패했습니다. "
                    "원문 접속 확인이 필요할 수 있습니다."
                )
            else:
                item["url"] = ""
                item["reliability_note"] = (
                    "유효한 원문 URL을 확인하지 못했습니다. DB 저장에서 제외됩니다."
                )

    return news_data


def safe_int(value, default=None):
    try:
        number = int(value)
        return max(1, min(5, number))
    except Exception:
        return default


def normalize_news_item(item, keyword):
    """Gemini 응답 데이터를 Supabase insert에 맞는 형태로 정리합니다."""
    tags = item.get("tags", "")
    if isinstance(tags, list):
        tags = ", ".join([str(tag).strip() for tag in tags if str(tag).strip()])

    importance_score = safe_int(item.get("importance_score"))

    return {
        "keyword": keyword.strip(),
        "title": str(item.get("title", "")).strip(),
        "source": str(item.get("source", "")).strip(),
        "news_date": str(item.get("news_date", "")).strip(),
        "url": str(item.get("url", "")).strip(),
        "summary": str(item.get("summary", "")).strip(),
        "one_line_summary": str(item.get("one_line_summary", "")).strip(),
        "background": str(item.get("background", "")).strip(),
        "why_it_matters": str(item.get("why_it_matters", "")).strip(),
        "stakeholders": str(item.get("stakeholders", "")).strip(),
        "impact_analysis": str(item.get("impact_analysis", "")).strip(),
        "issue_points": str(item.get("issue_points", "")).strip(),
        "sentiment": str(item.get("sentiment", "")).strip(),
        "importance_score": importance_score,
        "tags": tags,
        "source_type": str(item.get("source_type", "")).strip(),
        "reliability_note": str(item.get("reliability_note", "")).strip(),
    }


def build_news_prompt(keyword: str) -> str:
    return f"""
너는 최신 뉴스 검색과 비즈니스/사회적 함의 분석에 능숙한 AI 뉴스 애널리스트야.

검색 키워드: "{keyword}"

요구사항:
1. Google 검색을 사용해서 "{keyword}"에 대한 가장 최신 뉴스 딱 2건만 찾아.
2. 실제 기사에 근거해서 작성하고, 절대 URL을 지어내지 마.
3. 같은 사건을 제목만 다르게 다룬 중복성 기사는 피하고, 가능하면 서로 다른 관점의 기사 2건을 골라.
4. 응답은 반드시 아래 JSON 배열 형식으로만 작성해. 백틱(```), 설명 문장, 마크다운 없이 JSON만 출력해.
5. sentiment는 "긍정", "부정", "중립" 중 하나로 작성해.
6. importance_score는 1~5 사이의 정수로 작성해.
7. tags는 쉼표로 구분된 문자열로 작성해.

[
  {{
    "title": "기사 제목",
    "source": "언론사 또는 출처",
    "news_date": "기사 발행일. 가능하면 YYYY-MM-DD",
    "url": "기사 원본 URL",
    "summary": "기사 내용을 3문장 이내로 요약",
    "one_line_summary": "핵심을 한 문장으로 압축",
    "background": "이 뉴스가 나오게 된 배경 맥락",
    "why_it_matters": "이 뉴스가 중요한 이유",
    "stakeholders": "주요 이해관계자. 예: 기업, 정부, 소비자, 투자자 등",
    "impact_analysis": "산업, 시장, 정책, 소비자, 기술 측면에서 예상되는 영향",
    "issue_points": "쟁점, 논란, 리스크, 비판 가능성",
    "sentiment": "긍정/부정/중립 중 하나",
    "importance_score": 1,
    "tags": "핵심 키워드1, 핵심 키워드2, 핵심 키워드3",
    "source_type": "언론사/공식기관/기업/기타 중 하나",
    "reliability_note": "출처 및 URL 확인 관련 참고 메모"
  }}
]
"""


@st.cache_data(ttl=60)
def fetch_news_history():
    response = (
        supabase
        .table("news_history")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data


def prepare_dataframe(data):
    df = pd.DataFrame(data)

    if df.empty:
        return df

    # 새 컬럼이 없는 과거 DB/과거 데이터에도 화면이 깨지지 않도록 빈 컬럼을 보정합니다.
    for col in ANALYSIS_COLUMNS:
        if col not in df.columns:
            df[col] = None

    if "created_at" in df.columns:
        df["created_at_datetime"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["created_at"] = df["created_at_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    if "importance_score" in df.columns:
        df["importance_score"] = pd.to_numeric(df["importance_score"], errors="coerce")

    return df


def display_news_card(item, idx):
    sentiment = item.get("sentiment") or "미분류"
    importance = item.get("importance_score") or "미분류"
    tags = item.get("tags") or ""

    with st.container(border=True):
        st.markdown(f"### {idx + 1}. [{item.get('title', '제목 없음')}]({item.get('url', '')})")
        st.caption(
            f"출처: {item.get('source', '미상')} | "
            f"날짜: {item.get('news_date', '미상')} | "
            f"감성: {sentiment} | "
            f"중요도: {importance}/5"
        )

        if item.get("one_line_summary"):
            st.markdown(f"**핵심 한 줄:** {item.get('one_line_summary')}")

        st.write(f"**요약:** {item.get('summary', '')}")

        with st.expander("🧠 AI 심층 분석 보기", expanded=True):
            if item.get("background"):
                st.markdown(f"**배경 맥락:** {item.get('background')}")
            if item.get("why_it_matters"):
                st.markdown(f"**왜 중요한가:** {item.get('why_it_matters')}")
            if item.get("stakeholders"):
                st.markdown(f"**주요 이해관계자:** {item.get('stakeholders')}")
            if item.get("impact_analysis"):
                st.markdown(f"**예상 영향:** {item.get('impact_analysis')}")
            if item.get("issue_points"):
                st.markdown(f"**쟁점/리스크:** {item.get('issue_points')}")
            if tags:
                st.markdown(f"**태그:** {tags}")
            if item.get("reliability_note"):
                st.caption(f"🔗 URL 확인: {item.get('reliability_note')}")


# ----------------------------------------------------
# 3. 화면 구성
# ----------------------------------------------------
st.title("📰 AI 최신 뉴스 검색 & 인사이트 자동 저장기")
st.markdown(
    "키워드를 검색하면 Gemini가 Google 검색으로 최신 뉴스 2건을 찾고, "
    "요약·중요도·감성·쟁점·예상 영향까지 분석해 Supabase에 저장합니다."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 검색하기",
    "💾 저장된 뉴스 보기",
    "📊 통계 분석",
    "🧠 AI 인사이트 리포트"
])


# ==========================================
# Tab 1: 검색 및 저장
# ==========================================
with tab1:
    st.subheader("새로운 뉴스 검색")

    st.info(
        "처음 수정본을 적용하는 경우, 먼저 Supabase SQL Editor에서 "
        "`supabase_news_history_migration.sql`을 실행해 주세요."
    )

    with st.form("search_form"):
        keyword = st.text_input("검색할 키워드를 입력하세요", placeholder="예: 생성형 AI, 테슬라, 한국경제, 반도체")
        submitted = st.form_submit_button("검색·분석·저장하기 🚀")

    if submitted and keyword.strip():
        with st.spinner(f"'{keyword}'에 대한 최신 뉴스를 검색하고 심층 분석 중입니다..."):
            try:
                prompt = build_news_prompt(keyword)

                # Google Search Tool과 response_mime_type은 동시에 사용하지 않습니다.
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        tools=[{"google_search": {}}],
                    ),
                )

                news_data = parse_json_array(response.text)
                grounding_links = extract_grounding_links(response)
                news_data = overwrite_urls_with_grounding(news_data, grounding_links)

                saved_count = 0
                skipped_count = 0
                invalid_url_count = 0

                st.success("✨ 검색과 분석이 완료되었습니다!")

                for idx, raw_item in enumerate(news_data[:2]):
                    item = normalize_news_item(raw_item, keyword)
                    display_news_card(item, idx)

                    if not is_real_http_url(item["url"]):
                        invalid_url_count += 1
                        st.warning(f"{idx + 1}번 뉴스는 유효한 원문 URL을 확인하지 못해 저장하지 않았습니다.")
                        continue

                    try:
                        supabase.table("news_history").insert(item).execute()
                        saved_count += 1
                    except Exception as e:
                        error_text = str(e)
                        if "23505" in error_text or "duplicate key" in error_text.lower():
                            skipped_count += 1
                        elif "Could not find" in error_text or "schema cache" in error_text:
                            st.error(
                                "DB 컬럼이 아직 추가되지 않은 것 같습니다. "
                                "Supabase SQL Editor에서 migration SQL을 먼저 실행해 주세요."
                            )
                            st.code(error_text)
                        else:
                            st.error(f"DB 저장 중 오류 발생: {e}")

                fetch_news_history.clear()
                st.toast(
                    f"✅ 새 뉴스 {saved_count}건 저장 완료! "
                    f"중복 생략 {skipped_count}건, URL 미확인 {invalid_url_count}건",
                    icon="🎉",
                )

            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                with st.expander("문제 해결 힌트"):
                    st.markdown(
                        """
                        - Gemini 응답이 JSON 배열 형식이 아닐 수 있습니다.
                        - Supabase 컬럼 추가 SQL을 아직 실행하지 않았을 수 있습니다.
                        - Streamlit Secrets의 키 이름이 정확한지 확인해 주세요.
                        """
                    )


# ==========================================
# Tab 2: 저장된 뉴스 보기
# ==========================================
with tab2:
    st.subheader("💾 DB에 저장된 뉴스 히스토리")

    data = fetch_news_history()

    if data:
        df = prepare_dataframe(data)

        filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])

        with filter_col1:
            filter_text = st.text_input("🔍 제목·키워드·요약·태그 검색", placeholder="예: 규제, AI, 투자")

        with filter_col2:
            sentiment_values = ["전체"] + sorted([
                str(x) for x in df["sentiment"].dropna().unique().tolist() if str(x).strip()
            ])
            selected_sentiment = st.selectbox("감성 필터", sentiment_values)

        with filter_col3:
            min_importance = st.slider("최소 중요도", 1, 5, 1)

        filtered_df = df.copy()

        if filter_text:
            search_cols = ["title", "keyword", "summary", "tags", "source"]
            mask = False
            for col in search_cols:
                if col in filtered_df.columns:
                    mask = mask | filtered_df[col].astype(str).str.contains(
                        filter_text,
                        case=False,
                        na=False,
                    )
            filtered_df = filtered_df[mask]

        if selected_sentiment != "전체":
            filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

        if "importance_score" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["importance_score"].fillna(0) >= min_importance
            ]

        st.caption(f"현재 조건에 맞는 뉴스: {len(filtered_df)}건")

        display_columns = [
            "keyword",
            "title",
            "source",
            "news_date",
            "sentiment",
            "importance_score",
            "tags",
            "one_line_summary",
            "why_it_matters",
            "impact_analysis",
            "url",
            "created_at",
        ]

        available_columns = [col for col in display_columns if col in filtered_df.columns]

        st.dataframe(
            filtered_df[available_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "url": st.column_config.LinkColumn("url"),
                "importance_score": st.column_config.NumberColumn("importance_score", format="%d점"),
            },
        )

        csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 현재 화면 데이터 CSV로 다운로드",
            data=csv,
            file_name=f"news_insight_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    else:
        st.info("아직 저장된 뉴스가 없습니다. 탭 1에서 뉴스를 검색해 보세요!")


# ==========================================
# Tab 3: 통계 대시보드
# ==========================================
with tab3:
    st.subheader("📊 뉴스 수집 통계 대시보드")

    data = fetch_news_history()

    if data:
        stat_df = prepare_dataframe(data)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("총 저장 뉴스", f"{len(stat_df)}건")
        with metric_col2:
            st.metric("검색 키워드 수", f"{stat_df['keyword'].nunique()}개")
        with metric_col3:
            avg_importance = stat_df["importance_score"].dropna().mean()
            st.metric("평균 중요도", f"{avg_importance:.2f}/5" if pd.notna(avg_importance) else "데이터 없음")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📌 키워드별 누적 수집 건수")
            keyword_counts = stat_df["keyword"].value_counts()
            st.bar_chart(keyword_counts)

        with col2:
            st.markdown("##### 📅 일자별 뉴스 저장 건수")
            if "created_at_datetime" in stat_df.columns:
                stat_df["date_only"] = stat_df["created_at_datetime"].dt.strftime("%Y-%m-%d")
                date_counts = stat_df["date_only"].value_counts().sort_index()
                st.line_chart(date_counts)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("##### 😊 감성 분석 비율")
            sentiment_counts = stat_df["sentiment"].fillna("미분류").replace("", "미분류").value_counts()
            st.bar_chart(sentiment_counts)

        with col4:
            st.markdown("##### 📰 출처별 저장 건수 TOP 10")
            source_counts = stat_df["source"].fillna("미상").replace("", "미상").value_counts().head(10)
            st.bar_chart(source_counts)

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("##### ⭐ 키워드별 평균 중요도")
            importance_by_keyword = (
                stat_df
                .dropna(subset=["importance_score"])
                .groupby("keyword")["importance_score"]
                .mean()
                .sort_values(ascending=False)
            )
            if not importance_by_keyword.empty:
                st.bar_chart(importance_by_keyword)
            else:
                st.info("중요도 데이터가 아직 없습니다.")

        with col6:
            st.markdown("##### 🏷️ 태그 빈도 TOP 10")
            tag_counter = Counter()

            for tags in stat_df["tags"].dropna():
                for tag in str(tags).split(","):
                    cleaned_tag = tag.strip()
                    if cleaned_tag:
                        tag_counter[cleaned_tag] += 1

            if tag_counter:
                tag_df = pd.DataFrame(
                    tag_counter.most_common(10),
                    columns=["tag", "count"],
                ).set_index("tag")
                st.bar_chart(tag_df)
            else:
                st.info("태그 데이터가 아직 없습니다.")

    else:
        st.info("통계를 표시할 데이터가 부족합니다.")


# ==========================================
# Tab 4: 저장된 뉴스 기반 AI 리포트
# ==========================================
with tab4:
    st.subheader("🧠 저장된 뉴스 기반 AI 인사이트 리포트")

    data = fetch_news_history()

    if data:
        report_df = prepare_dataframe(data).head(10)

        st.markdown(
            "최근 저장된 뉴스 최대 10건을 바탕으로, "
            "주요 흐름·기회·리스크·추가로 볼 키워드를 자동 정리합니다."
        )

        if st.button("AI 인사이트 리포트 생성하기 ✍️"):
            with st.spinner("저장된 뉴스를 종합해 리포트를 작성 중입니다..."):
                news_brief = report_df[
                    [
                        "keyword",
                        "title",
                        "source",
                        "summary",
                        "one_line_summary",
                        "why_it_matters",
                        "impact_analysis",
                        "issue_points",
                        "sentiment",
                        "importance_score",
                        "tags",
                    ]
                ].fillna("").to_dict(orient="records")

                report_prompt = f"""
아래는 사용자가 최근 저장한 뉴스 데이터야.

이 데이터를 바탕으로 한국어 리포트를 작성해 줘.
단순 요약이 아니라, 전체 흐름을 분석하고 고차원적인 인사이트를 제시해.

작성 형식:
1. 전체 핵심 흐름 3가지
2. 특히 중요해 보이는 이슈
3. 긍정적 기회 요인
4. 리스크/논란 요인
5. 앞으로 추가로 검색하면 좋은 키워드 5개
6. 발표나 보고서에 쓸 수 있는 한 문장 결론

뉴스 데이터:
{json.dumps(news_brief, ensure_ascii=False)}
"""

                report_response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=report_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                    ),
                )

                st.markdown(report_response.text)

    else:
        st.info("리포트를 만들려면 먼저 뉴스를 검색하고 저장해 주세요.")
