import streamlit as st
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

# Streamlit Cloud用: ファイル監視を無効化
os.environ["STREAMLIT_SERVER_WATCHER_TYPE"] = "none"

# .envファイルから環境変数を読み込み（ローカルのみ）
try:
    load_dotenv()
except:
    pass  # Cloud環境では無視

# APIキーの設定
gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEYが設定されていません。")
    st.info("ローカル: .envファイルに設定\nStreamlit Cloud: Settings → Secrets で設定してください")
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

# ===== 検索担当エージェント（キャッシュ対応版） =====
class SearchAgent:
    def __init__(self, llm):
        self.llm = llm
        self.search_cache = {}  # 検索結果のキャッシュ
    
    def generate_search_queries(self, query: str) -> List[str]:
        """質問から複数の検索クエリを生成（キャッシュ対応）"""
        # キャッシュを確認
        cache_key = query.lower().strip()
        
        if cache_key in self.search_cache:
            print(f"キャッシュヒット: {cache_key}")
            return self.search_cache[cache_key]
        
        # 新規クエリ生成
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
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                queries = data.get("queries", [query])
                
                # キャッシュに保存
                self.search_cache[cache_key] = queries
                
                return queries
            else:
                return [query]
        except:
            return [query]
    
    def get_cached_results(self, query: str) -> List[Dict[str, Any]]:
        """キャッシュから検索結果を取得"""
        cache_key = query.lower().strip()
        return self.search_cache.get(cache_key, [])
    
    def cache_search_result(self, query: str, result: str) -> None:
        """検索結果をキャッシュに保存"""
        cache_key = query.lower().strip()
        if cache_key not in self.search_cache:
            self.search_cache[cache_key] = []
        self.search_cache[cache_key].append({
            "query": query,
            "result": result,
            "source": "cached",
            "index": len(self.search_cache[cache_key]) - 1,
            "cached": True
        })
    
    def execute_searches(self, queries: List[str]) -> List[Dict[str, Any]]:
        """複数のクエリで検索を実行（キャッシュ対応）"""
        results = []
        cache_hits = 0
        
        for i, query in enumerate(queries):
            # まずキャッシュを確認
            cached_results = self.get_cached_results(query)
            if cached_results:
                print(f"キャッシュヒット: {query} ({len(cached_results)}件)")
                cache_hits += 1
                results.extend(cached_results)
                continue
            
            # キャッシュにない場合のみ新規検索
            print(f"新規検索: {query}")
            search_result = web_search.invoke(query)
            
            if search_result:
                # 検索結果をキャッシュに保存
                self.cache_search_result(query, search_result)
                
                results.append({
                    "query": query,
                    "result": search_result,
                    "source": f"search_{i+1}",
                    "index": i,
                    "cached": False
                })
        
        # キャッシュ統計をログ出力
        print(f"キャッシュ統計: {cache_hits}/{len(queries)} ヒット")
        
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
    
    def evaluate_content_quality_batch(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """複数のコンテンツを一括で評価"""
        if not search_results:
            return []
        
        # 評価対象を整形
        evaluation_data = []
        for i, result in enumerate(search_results):
            result_text = result["result"]
            title_match = re.search(r'タイトル: ([^\n]+)', result_text)
            title = title_match.group(1) if title_match else ""
            content_match = re.search(r'内容: ([^\n]+)', result_text)
            content = content_match.group(1) if content_match else result_text
            
            # 短すぎるコンテンツは早期除外
            if len(content) < 100:
                evaluation_data.append({
                    "index": i,
                    "score": 0.3,
                    "reason": "コンテンツが短すぎる"
                })
                continue
            
            evaluation_data.append({
                "index": i,
                "title": title,
                "content": content[:200],  # 最初の200文字のみ使用
                "needs_evaluation": True
            })
        
        if not evaluation_data:
            return [{"score": 0.5, "reason": "評価対象なし"}]
        
        # バッチ評価プロンプト
        batch_prompt = f"""
        以下の情報の品質を1-10で評価してください。評価基準：
        - 事実に基づいているか
        - 専門性があるか  
        - 最新情報か
        - 客観的記述か
        
        評価対象:
        {chr(10).join([f"{i+1}. タイトル: {data['title']}\n内容: {data['content']}..." for i, data in enumerate(evaluation_data)])}
        
        JSON形式で回答（評価結果のリスト）:
        {{
            "evaluations": [
                {{"index": 0, "score": 0.8, "reason": "専門的で信頼性高い"}},
                {{"index": 1, "score": 0.6, "reason": "一般的な情報"}}
            ]
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=batch_prompt)])
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                evaluations = data.get("evaluations", [])
                
                # 結果を元の形式に戻す
                results = []
                for i, eval_data in enumerate(evaluation_data):
                    if i < len(evaluations):
                        score = min(evaluations[i].get("score", 0.5) / 10, 1.0)
                        reason = evaluations[i].get("reason", "AI評価")
                    else:
                        score = eval_data.get("score", 0.5)
                        reason = eval_data.get("reason", "評価不能")
                    
                    results.append({
                        "index": eval_data["index"],
                        "score": score,
                        "reason": reason
                    })
                
                return results
        except:
            # エラー時は個別評価
            return [
                {"score": 0.5, "reason": "バッチ評価失敗"}
                for _ in evaluation_data
            ]
    
    def evaluate_content_quality(self, title: str, content: str) -> Dict[str, Any]:
        """コンテンツの品質を客観的基準で評価"""
        evaluation_criteria = {
            "factual_basis": "事実との一致度（1-10）",
            "source_credibility": "公的情報源か（1-10）",
            "technical_accuracy": "技術的正確さ（1-10）",
            "information_depth": "情報の深さと網羅性（1-10）",
            "objectivity": "客観性（1-10）"
        }
        
        prompt = f"""
        以下のコンテンツを客観的基準で評価してください。
        
        評価対象:
        タイトル: {title}
        内容: {content[:200]}
        
        評価基準：
        1. 事実との一致度（1-10）：情報が検証可能な事実とどの程度一致しているか
        2. 公的情報源か（1-10）：政府機関、学術機関、信頼できるメディアなど
        3. 技術的正確さ（1-10）：専門用語の適切さ、データの正確さ、技術的な深さ
        4. 情報の深さと網羅性（1-10）：多角的な視点、情報の網羅性、包括性
        5. 客観性（1-10）：個人的な偏見の排除、中立的な記述
        
        各基準について1-10点で評価し、その根拠を簡潔に記述してください。
        
        JSON形式で回答：
        {{
            "factual_basis": 0-10のスコア,
            "source_credibility": 0-10のスコア,
            "technical_accuracy": 0-10のスコア,
            "information_depth": 0-10のスコア,
            "objectivity": 0-10のスコア,
            "total_score": 0-50のスコア,
            "evaluation_reason": "各基準の評価根拠"
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                total_score = min(data.get("total_score", 25) / 50, 1.0)  # 0-50の範囲に正規化
                
                return {
                    "score": total_score,
                    "reason": f"客観的評価: 総合{total_score*2:.1f}点",
                    "criteria_scores": {
                        "factual_basis": data.get("factual_basis", 5),
                        "source_credibility": data.get("source_credibility", 5),
                        "technical_accuracy": data.get("technical_accuracy", 5),
                        "information_depth": data.get("information_depth", 5),
                        "objectivity": data.get("objectivity", 5)
                    }
                }
        except:
            return {"score": 0.5, "reason": "客観的評価失敗"}
    
    def calculate_overall_reliability(self, search_result: Dict[str, Any], content_evaluations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        
        # バッチ評価結果を使用
        if content_evaluations:
            result_index = search_result.get("index", 0)
            if result_index < len(content_evaluations):
                content_score = content_evaluations[result_index]
            else:
                content_score = {"score": 0.5, "reason": "評価対象外"}
        else:
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
        # まずバッチ評価を実行
        content_evaluations = self.evaluate_content_quality_batch(search_results)
        
        reliability_scores = []
        filtered_results = []
        
        for i, result in enumerate(search_results):
            # 各評価を実行
            domain_score = self.evaluate_domain_reliability(
                self.extract_url_from_result(result["result"])
            )
            
            # バッチ評価結果を使用
            if i < len(content_evaluations):
                content_score = content_evaluations[i]
            else:
                content_score = {"score": 0.5, "reason": "評価対象外"}
            
            # 新鮮度評価（簡易的）
            freshness_score = {"score": 0.7, "reason": "検索結果"}
            
            # 重み付き平均（ドメイン40%、コンテンツ40%、新鮮度20%）
            overall_score = (
                domain_score["score"] * 0.4 +
                content_score["score"] * 0.4 +
                freshness_score["score"] * 0.2
            )
            
            score_info = {
                "overall_score": round(overall_score, 2),
                "domain_score": domain_score,
                "content_score": content_score,
                "freshness_score": freshness_score,
                "url": self.extract_url_from_result(result["result"]),
                "title": re.search(r'タイトル: ([^\n]+)', result["result"]).group(1) if re.search(r'タイトル: ([^\n]+)', result["result"]) else "",
                "recommendation": "高品質" if overall_score >= 0.7 else "使用可" if overall_score >= 0.5 else "低品質"
            }
            
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
def should_use_multi_agent_node(state: MultiAgentState) -> MultiAgentState:
    """マルチエージェントが必要か判断するノード"""
    decision = should_use_multi_agent(state)
    state["decision"] = decision
    return state

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
workflow.add_node("should_use_multi_agent", should_use_multi_agent_node)
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

st.title("AIリサーチ")
st.caption("情報の信頼性をAIが自動評価し、高品質な情報のみを使用したリサーチを実行")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="こんにちは！AIリサーチシステムです。低品質な情報を自動除外し、信頼性の高い情報のみを使用します！")
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
    - 複数の角度から情報収集
    
    **🔍 信頼性評価エージェント**  
    - コンテンツ品質評価
    - 自動フィルタリング
    - 情報源の信頼性評価

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
            AIMessage(content="こんにちは！AIリサーチシステムです。低品質な情報を自動除外し、信頼性の高い情報のみを使用します！")
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
