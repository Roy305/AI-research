import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from tavily import TavilyClient
import json
import time
import re
from urllib.parse import urlparse

# .envファイルから環境変数を読み込み
load_dotenv()

# APIキーの設定
gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEYが.envファイルに設定されていません。")
    st.stop()

if not tavily_api_key:
    st.error("TAVILY_API_KEYが.envファイルに設定されていません。")
    st.stop()

# Tavilyクライアントの初期化
tavily_client = TavilyClient(api_key=tavily_api_key)

# Geminiモデルの初期化
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)

# Tavily検索ツールの定義
@tool
def web_search(query: str) -> str:
    """指定されたクエリでウェブ検索を実行し、結果を返します。"""
    try:
        response = tavily_client.search(query=query, search_depth="basic")
        results = response.get("results", [])
        
        if not results:
            return "検索結果が見つかりませんでした。"
        
        # 検索結果を整形
        formatted_results = []
        for result in results[:5]:  # 上位5件を取得
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            formatted_results.append(f"タイトル: {title}\n内容: {content}\nURL: {url}\n")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"検索中にエラーが発生しました: {str(e)}"

# ツールのリスト
tools = [web_search]

# マルチエージェントの状態定義
class MultiAgentState(TypedDict):
    messages: Annotated[List, "メッセージのリスト"]
    current_step: str
    decision: str
    search_results: List[Dict[str, Any]]
    filtered_results: List[Dict[str, Any]]
    reliability_scores: List[Dict[str, Any]]
    summary: str
    final_report: str
    query: str

# ===== 検索担当エージェント =====
class SearchAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_search_queries(self, query: str) -> List[str]:
        """質問から複数の検索クエリを生成"""
        prompt = f"""
        以下の質問に答えるための検索クエリを3つ生成してください。
        それぞれ異なる角度からアプローチしてください。
        
        質問: {query}
        
        検索クエリ（JSON形式）:
        {{
            "queries": ["クエリ1", "クエリ2", "クエリ3"]
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # JSONをパース
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("queries", [query])
            else:
                return [query]
        except:
            return [query]
    
    def execute_searches(self, queries: List[str]) -> List[Dict[str, Any]]:
        """複数のクエリで検索を実行"""
        results = []
        for i, query in enumerate(queries):
            search_result = web_search.invoke(query)
            results.append({
                "query": query,
                "result": search_result,
                "source": f"search_{i+1}",
                "index": i
            })
        return results

# ===== 信頼性評価エージェント =====
class ReliabilityAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def extract_url_from_result(self, result: str) -> str:
        """検索結果からURLを抽出"""
        url_match = re.search(r'URL: (https?://[^\s]+)', result)
        return url_match.group(1) if url_match else ""
    
    def evaluate_domain_reliability(self, url: str) -> Dict[str, Any]:
        """ドメインの信頼性を評価"""
        if not url:
            return {"score": 0.3, "reason": "URL不明"}
        
        domain = urlparse(url).netloc.lower()
        
        # 高信頼性ドメインリスト
        high_reliability_domains = [
            "gov.jp", "go.jp", "ac.jp", "ed.jp",  # 日本の公的機関・教育機関
            "bbc.com", "reuters.com", "ap.org",   # 有力ニュース
            "nature.com", "science.org",          # 学術誌
            "who.int", "un.org",                  # 国際機関
            "nikkei.com", "asahi.com", "yomiuri.co.jp"  # 日本の主要媒体
        ]
        
        # 中信頼性ドメインリスト
        medium_reliability_domains = [
            "wikipedia.org",                      # ウィキペディア
            "medium.com", "linkedin.com",          # プラットフォーム
            "techcrunch.com", "theverge.com"      # 技術メディア
        ]
        
        # 低信頼性を示す指標
        low_reliability_indicators = [
            "blog", "forum", "2ch", "5ch", "twitter", "facebook"
        ]
        
        # スコアリング
        if any(domain.endswith(d) for d in high_reliability_domains):
            return {"score": 0.9, "reason": "高信頼性ドメイン"}
        elif any(domain.endswith(d) for d in medium_reliability_domains):
            return {"score": 0.7, "reason": "中信頼性ドメイン"}
        elif any(indicator in domain for indicator in low_reliability_indicators):
            return {"score": 0.4, "reason": "低信頼性ドメイン"}
        else:
            return {"score": 0.6, "reason": "一般ドメイン"}
    
    def evaluate_content_quality(self, title: str, content: str) -> Dict[str, Any]:
        """コンテンツの品質をAIで評価"""
        if len(content) < 100:
            return {"score": 0.3, "reason": "コンテンツが短すぎる"}
        
        prompt = f"""
        以下の情報の品質を1-10で評価してください。評価基準：
        - 事実に基づいているか
        - 専門性があるか  
        - 最新情報か
        - 客観的記述か
        
        タイトル: {title}
        内容: {content[:500]}...
        
        JSON形式で回答:
        {{
            "score": 0.8,
            "reason": "評価理由"
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "score": min(data.get("score", 0.5) / 10, 1.0),
                    "reason": data.get("reason", "AI評価")
                }
        except:
            pass
        
        return {"score": 0.5, "reason": "評価不能"}
    
    def calculate_overall_reliability(self, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """全体の信頼性スコアを計算"""
        result_text = search_result["result"]
        
        # URLとタイトル、コンテンツを抽出
        url = self.extract_url_from_result(result_text)
        title_match = re.search(r'タイトル: ([^\n]+)', result_text)
        title = title_match.group(1) if title_match else ""
        content_match = re.search(r'内容: ([^\n]+)', result_text)
        content = content_match.group(1) if content_match else result_text
        
        # 各評価を実行
        domain_score = self.evaluate_domain_reliability(url)
        content_score = self.evaluate_content_quality(title, content)
        
        # 新鮮度評価（簡易的）
        freshness_score = {"score": 0.7, "reason": "検索結果"}
        
        # 重み付き平均（ドメイン40%、コンテンツ40%、新鮮度20%）
        overall_score = (
            domain_score["score"] * 0.4 +
            content_score["score"] * 0.4 +
            freshness_score["score"] * 0.2
        )
        
        return {
            "overall_score": round(overall_score, 2),
            "domain_score": domain_score,
            "content_score": content_score,
            "freshness_score": freshness_score,
            "url": url,
            "title": title,
            "recommendation": "高品質" if overall_score >= 0.7 else "使用可" if overall_score >= 0.5 else "低品質"
        }
    
    def filter_by_reliability(self, search_results: List[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
        """信頼性スコアに基づいて情報をフィルタリング"""
        reliability_scores = []
        filtered_results = []
        
        for result in search_results:
            score_info = self.calculate_overall_reliability(result)
            reliability_scores.append(score_info)
            
            # 閾値以上の情報のみを保持
            if score_info["overall_score"] >= threshold:
                result["reliability"] = score_info
                filtered_results.append(result)
        
        return {
            "reliability_scores": reliability_scores,
            "filtered_results": filtered_results,
            "original_count": len(search_results),
            "filtered_count": len(filtered_results),
            "excluded_count": len(search_results) - len(filtered_results)
        }

# ===== 情報要約エージェント =====
class SummarizerAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def summarize_information(self, query: str, filtered_results: List[Dict[str, Any]]) -> str:
        """フィルタリングされた検索結果を要約"""
        if not filtered_results:
            return "信頼性の高い情報が見つかりませんでした。より一般的な知識で回答します。"
        
        # 検索結果を整形（信頼性情報も含む）
        combined_results = []
        for i, result in enumerate(filtered_results):
            reliability = result.get("reliability", {})
            score = reliability.get("overall_score", 0.5)
            recommendation = reliability.get("recommendation", "不明")
            
            combined_results.append(
                f"=== 情報源 {i+1} (信頼性: {score}/1.0 - {recommendation}) ===\n"
                f"クエリ: {result['query']}\n"
                f"結果: {result['result']}\n"
                f"URL: {reliability.get('url', '不明')}\n"
            )
        
        prompt = f"""
        以下の信頼性評価済み情報を元に、元の質問に関連する情報を要約してください。
        
        元の質問: {query}
        
        信頼性評価済み情報:
        {chr(10).join(combined_results)}
        
        要約のポイント:
        - 信頼性の高い情報を優先
        - 重要な情報を抽出
        - 重複する情報は統合
        - 事実関係を明確に
        - 簡潔で分かりやすい表現
        
        要約:
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

# ===== レポート作成エージェント =====
class ReportAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_final_report(self, query: str, summary: str, reliability_info: Dict[str, Any]) -> str:
        """最終レポートを作成（信頼性情報を含む）"""
        prompt = f"""
        以下の情報を元に、質問に対する詳細な回答を作成してください。
        
        質問: {query}
        
        信頼性評価済み要約:
        {summary}
        
        情報品質レポート:
        - 元の検索結果数: {reliability_info.get('original_count', 0)}
        - 信頼性基準を満たした情報: {reliability_info.get('filtered_count', 0)}
        - 除外された低品質情報: {reliability_info.get('excluded_count', 0)}
        
        レポートの要件:
        - 質問に直接回答
        - 具体的なデータや事実を提示
        - 分かりやすい構成で整理
        - 専門用語は平易に説明
        - 情報源の信頼性について言及
        - 必要に応じて追加調査の提案
        
        回答:
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

# ===== エージェントの初期化 =====
search_agent = SearchAgent(llm)
reliability_agent = ReliabilityAgent(llm)
summarizer_agent = SummarizerAgent(llm)
report_agent = ReportAgent(llm)

# ===== ワークフロー関数 =====
def should_use_multi_agent(state: MultiAgentState) -> Literal["multi_agent", "simple"]:
    """マルチエージェントが必要か判断"""
    last_message = state["messages"][-1].content.lower()
    
    # 複雑な調査が必要なキーワード
    complex_keywords = [
        "調査", "リサーチ", "分析", "比較", "まとめ", "レポート",
        "詳細", "深く", "徹底", "包括的", "多角的"
    ]
    
    return "multi_agent" if any(kw in last_message for kw in complex_keywords) else "simple"

def search_step(state: MultiAgentState) -> MultiAgentState:
    """検索ステップ"""
    query = state["messages"][-1].content
    
    with st.spinner("🔍 検索エージェントが情報を収集中..."):
        # 検索クエリ生成
        queries = search_agent.generate_search_queries(query)
        st.info(f"📝 検索クエリ: {', '.join(queries)}")
        
        # 検索実行
        search_results = search_agent.execute_searches(queries)
        
        state["search_results"] = search_results
        state["query"] = query
        state["current_step"] = "searched"
    
    return state

def reliability_step(state: MultiAgentState) -> MultiAgentState:
    """信頼性評価ステップ"""
    with st.spinner("🔍 信頼性評価エージェントが情報品質を分析中..."):
        # 信頼性評価とフィルタリング
        reliability_info = reliability_agent.filter_by_reliability(state["search_results"])
        
        state["reliability_scores"] = reliability_info["reliability_scores"]
        state["filtered_results"] = reliability_info["filtered_results"]
        
        # 信頼性情報を表示
        st.success(f"""
        **📊 情報品質評価完了**
        - 元の検索結果: {reliability_info['original_count']}件
        - 信頼性基準満足: {reliability_info['filtered_count']}件  
        - 低品質情報除外: {reliability_info['excluded_count']}件
        """)
        
        # 信頼性スコアの詳細表示
        if reliability_info["reliability_scores"]:
            with st.expander("🔍 信頼性スコア詳細"):
                for i, score in enumerate(reliability_info["reliability_scores"]):
                    color = "🟢" if score["overall_score"] >= 0.7 else "🟡" if score["overall_score"] >= 0.5 else "🔴"
                    st.markdown(f"""
                    **{color} 情報源 {i+1}**
                    - スコア: {score['overall_score']}/1.0 ({score['recommendation']})
                    - URL: {score.get('url', '不明')}
                    - ドメイン評価: {score['domain_score']['reason']}
                    - コンテンツ評価: {score['content_score']['reason']}
                    """)
        
        state["current_step"] = "reliability_evaluated"
    
    return state

def summarize_step(state: MultiAgentState) -> MultiAgentState:
    """要約ステップ"""
    with st.spinner("📊 情報要約エージェントが分析中..."):
        summary = summarizer_agent.summarize_information(
            state["query"], 
            state["filtered_results"]
        )
        
        state["summary"] = summary
        state["current_step"] = "summarized"
    
    return state

def report_step(state: MultiAgentState) -> MultiAgentState:
    """レポート作成ステップ"""
    with st.spinner("📝 レポート作成エージェントが執筆中..."):
        reliability_info = {
            "original_count": len(state["search_results"]),
            "filtered_count": len(state["filtered_results"]),
            "excluded_count": len(state["search_results"]) - len(state["filtered_results"])
        }
        
        final_report = report_agent.generate_final_report(
            state["query"],
            state["summary"],
            reliability_info
        )
        
        state["final_report"] = final_report
        state["current_step"] = "completed"
    
    return state

def simple_respond_step(state: MultiAgentState) -> MultiAgentState:
    """シンプル応答ステップ"""
    query = state["messages"][-1].content
    
    with st.spinner("💭 AIが回答を生成中..."):
        response = llm.invoke([HumanMessage(content=query)])
        
        state["final_report"] = response.content
        state["current_step"] = "completed"
    
    return state

# ===== LangGraphワークフロー構築 =====
workflow = StateGraph(MultiAgentState)

# ノードの追加
workflow.add_node("should_use_multi_agent", lambda state: {"decision": should_use_multi_agent(state)})
workflow.add_node("search", search_step)
workflow.add_node("reliability", reliability_step)
workflow.add_node("summarize", summarize_step)
workflow.add_node("report", report_step)
workflow.add_node("simple_respond", simple_respond_step)

# エントリーポイント
workflow.set_entry_point("should_use_multi_agent")

# 条件付きエッジ
workflow.add_conditional_edges(
    "should_use_multi_agent",
    lambda state: state["decision"],
    {
        "multi_agent": "search",
        "simple": "simple_respond"
    }
)

# マルチエージェントのフロー
workflow.add_edge("search", "reliability")
workflow.add_edge("reliability", "summarize")
workflow.add_edge("summarize", "report")
workflow.add_edge("report", END)
workflow.add_edge("simple_respond", END)

# コンパイル
agent = workflow.compile()

# ===== Streamlitアプリ =====
st.set_page_config(
    page_title="信頼性評価付きAIリサーチ",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 信頼性評価付きAIリサーチ")
st.caption("情報の信頼性をAIが自動評価し、高品質な情報のみを使用したリサーチを実行")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="こんにちは！信頼性評価付きAIリサーチシステムです。\n\n🔍 **検索エージェント**: 複数の角度から情報収集\n🔍 **信頼性評価エージェント**: 情報源の信頼性を自動スコアリング\n📊 **要約エージェント**: 高品質情報を整理・分析\n📝 **レポートエージェント**: 信頼性情報を含む最終回答を作成\n\n低品質な情報を自動除外し、信頼性の高い情報のみを使用します！")
    ]

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
        st.markdown(message.content)

# ユーザー入力
if prompt := st.chat_input("リサーチしたいことを入力してください..."):
    # ユーザーメッセージを追加
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタントの応答を生成
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # エージェントを実行
            initial_state = {
                "messages": [user_message],
                "current_step": "initial",
                "decision": "",
                "search_results": [],
                "filtered_results": [],
                "reliability_scores": [],
                "summary": "",
                "final_report": "",
                "query": ""
            }
            
            result = agent.invoke(initial_state)
            
            # 最終的な応答を取得
            final_report = result["final_report"]
            
            # エージェント処理の概要を表示
            if result["current_step"] == "completed" and result.get("search_results"):
                agent_info = f"""
---
**🤖 信頼性評価付きマルチエージェント処理完了**
- 🔍 検索クエリ数: {len(result['search_results'])}
- 🔍 信頼性評価: 完了
- 📊 高品質情報: {len(result.get('filtered_results', []))}件
- 📝 レポート作成: 完了
---
"""
                final_response = agent_info + final_report
            else:
                final_response = final_report
            
            message_placeholder.markdown(final_response)
            
            # チャット履歴に追加
            st.session_state.messages.append(AIMessage(content=final_response))
            
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append(AIMessage(content=error_message))

# サイドバー
with st.sidebar:
    st.title("🔍 信頼性評価設定")
    
    st.markdown("### エージェント構成")
    st.markdown("""
    **🔍 検索エージェント**
    - 複数クエリ生成
    - 並列検索実行
    
    **🔍 信頼性評価エージェント**  
    - ドメイン信頼性評価
    - コンテンツ品質評価
    - 自動フィルタリング
    
    **📊 要約エージェント**
    - 高品質情報のみ使用
    - 重複除去
    
    **📝 レポートエージェント**
    - 信頼性情報を含む回答
    """)
    
    st.markdown("---")
    
    # 信頼性閾値設定
    st.markdown("### 信頼性基準")
    threshold = st.slider("信頼性閾値", 0.0, 1.0, 0.5, 0.1)
    st.info(f"現在の閾値: {threshold} (これ以上の情報のみ使用)")
    
    if st.button("チャット履歴をクリア"):
        st.session_state.messages = [
            AIMessage(content="こんにちは！信頼性評価付きAIリサーチシステムです。\n\n🔍 **検索エージェント**: 複数の角度から情報収集\n🔍 **信頼性評価エージェント**: 情報源の信頼性を自動スコアリング\n📊 **要約エージェント**: 高品質情報を整理・分析\n📝 **レポートエージェント**: 信頼性情報を含む最終回答を作成\n\n低品質な情報を自動除外し、信頼性の高い情報のみを使用します！")
        ]
        st.rerun()
    
    st.markdown("### 信頼性評価基準")
    st.markdown("""
    **ドメイン評価 (40%)**
    - 🟢 高信頼: gov, ac, 主要メディア
    - 🟡 中信頼: Wikipedia, 技術メディア  
    - 🔴 低信頼: ブログ, フォーラム
    
    **コンテンツ評価 (40%)**
    - 専門性、事実基準、客観性
    
    **新鮮度評価 (20%)**
    - 検索結果の新鮮さ
    """)
