import os
import json
import hashlib
import requests
import feedparser
from datetime import datetime, timezone
from anthropic import Anthropic

# ── 配置 ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_WSJ_DIGEST_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

SEEN_IDS_FILE = "seen_ids.json"
MAX_SEEN_IDS = 2000       # 防止文件无限增长
MIN_SCORE = 7             # 低于此分数不推送

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── 数据源 ────────────────────────────────────────────
RSS_FEEDS = [
    {
        "name": "CNBC Markets",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"
    },
    {
        "name": "Bloomberg Markets",
        "url": "https://feeds.bloomberg.com/markets/news.rss"
    },
    {
        "name": "MarketWatch",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/"
    },
]
# ── 读写去重记录 ───────────────────────────────────────
def load_seen_ids() -> set:
    if os.path.exists(SEEN_IDS_FILE):
        with open(SEEN_IDS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_seen_ids(seen: set):
    ids = list(seen)
    if len(ids) > MAX_SEEN_IDS:
        ids = ids[-MAX_SEEN_IDS:]
    with open(SEEN_IDS_FILE, "w") as f:
        json.dump(ids, f)

def make_id(title: str, source: str) -> str:
    return hashlib.md5(f"{source}::{title}".encode()).hexdigest()

# ── 抓取新闻 ──────────────────────────────────────────
def fetch_rss(feed: dict) -> list[dict]:
    articles = []
    try:
        parsed = feedparser.parse(feed["url"])
        for entry in parsed.entries[:15]:
            articles.append({
                "id": make_id(entry.get("title", ""), feed["name"]),
                "title": entry.get("title", "").strip(),
                "summary": entry.get("summary", "")[:300].strip(),
                "source": feed["name"],
                "url": entry.get("link", ""),
            })
    except Exception as e:
        print(f"[RSS error] {feed['name']}: {e}")
    return articles

def fetch_finnhub() -> list[dict]:
    if not FINNHUB_API_KEY:
        return []
    articles = []
    try:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": "general", "token": FINNHUB_API_KEY}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        items = data if isinstance(data, list) else []
        for item in items[:20]:
            articles.append({
                "id": make_id(item.get("headline", ""), "Finnhub"),
                "title": item.get("headline", "").strip(),
                "summary": item.get("summary", "")[:300].strip(),
                "source": item.get("source", "Finnhub"),
                "url": item.get("url", ""),
            })
    except Exception as e:
        print(f"[Finnhub error]: {e}")
    return articles

def fetch_all_news() -> list[dict]:
    all_articles = []
    for feed in RSS_FEEDS:
        all_articles.extend(fetch_rss(feed))
    all_articles.extend(fetch_finnhub())
    return all_articles

# ── Claude 批量分析 ───────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的美股市场信号分析师。

我会给你一批最新财经新闻，请分析每条新闻对美股的潜在影响。

评分标准（1-10分）：
- 9-10分：极强信号。美联储意外决策、重大地缘政治突发、市场系统性风险、知名CEO/投资人对具体股票的重磅表态
- 7-8分：强信号。宏观数据超预期、知名公司重大并购/财报、行业重要政策变化、大佬言论涉及具体板块
- 5-6分：中等信号。普通财报、常规分析师评级变化、行业会议一般性发言
- 1-4分：弱信号或无关信息

重要原则：
- 语义理解优先于关键词匹配，任何可能导致股价5%以上波动的信息都要高分
- 即使新闻未提及"Fed"、"CPI"等关键词，只要内容重要就给高分
- CEO或知名投资人对具体公司/行业的公开表态，视影响力给7-9分

请严格按以下JSON格式输出，不要输出任何其他内容：
[
  {
    "id": "新闻编号（从0开始）",
    "score": 数字,
    "direction": "bullish或bearish或neutral",
    "sectors": ["相关板块，如：半导体、能源、科技、金融等"],
    "tickers": ["相关股票代码，如NVDA、AAPL，没有则空数组"],
    "reason": "一句话说明为何给此分数和方向"
  }
]"""

def analyze_with_claude(articles: list[dict]) -> list[dict]:
    if not articles:
        return []

    # 构建新闻列表文本
    news_text = ""
    for i, a in enumerate(articles):
        news_text += f"\n[{i}] 来源：{a['source']}\n标题：{a['title']}\n摘要：{a['summary']}\n"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": f"请分析以下{len(articles)}条新闻：\n{news_text}"
            }],
            system=SYSTEM_PROMPT,
        )
        raw = response.content[0].text.strip()

        # 清理可能的markdown代码块
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        results = json.loads(raw)
        return results
    except Exception as e:
        print(f"[Claude error]: {e}")
        return []

# ── Telegram 推送 ─────────────────────────────────────
DIRECTION_EMOJI = {
    "bullish": "🟢",
    "bearish": "🔴",
    "neutral": "🟡",
}

SCORE_EMOJI = {
    9: "🔥🔥",
    8: "🔥",
    7: "⚡",
}

def format_message(article: dict, analysis: dict) -> str:
    score = analysis["score"]
    direction = analysis["direction"]
    sectors = "、".join(analysis.get("sectors", []))
    tickers = " ".join([f"${t}" for t in analysis.get("tickers", [])])
    reason = analysis.get("reason", "")

    score_icon = SCORE_EMOJI.get(score, "⚡") if score >= 9 else SCORE_EMOJI.get(score, "⚡")
    direction_icon = DIRECTION_EMOJI.get(direction, "🟡")

    lines = [
        f"{score_icon} *{score}/10* | {direction_icon} {direction.upper()}",
        f"",
        f"*{article['title']}*",
        f"",
        f"📌 {reason}",
    ]
    if sectors:
        lines.append(f"🏭 板块：{sectors}")
    if tickers:
        lines.append(f"📈 股票：{tickers}")
    lines.append(f"")
    lines.append(f"🔗 [原文]({article['url']}) | 来源：{article['source']}")

    return "\n".join(lines)

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if not res.ok:
            print(f"[Telegram error]: {res.text}")
    except Exception as e:
        print(f"[Telegram error]: {e}")

# ── 主流程 ────────────────────────────────────────────
def main():
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] 开始运行...")

    # 1. 加载已读记录
    seen_ids = load_seen_ids()

    # 2. 抓取所有新闻
    all_articles = fetch_all_news()
    print(f"抓取到 {len(all_articles)} 条新闻")

    # 3. 过滤已读
    new_articles = [a for a in all_articles if a["id"] not in seen_ids]
    print(f"其中新增 {len(new_articles)} 条")

    if not new_articles:
        print("无新增新闻，退出")
        return

    # 4. Claude批量分析
    results = analyze_with_claude(new_articles[:20])
    print(f"Claude分析完成，共 {len(results)} 条结果")

    # 5. 推送高分信号
    pushed = 0
    for r in results:
        idx = int(r.get("id", -1))
        score = r.get("score", 0)
        if score >= MIN_SCORE and 0 <= idx < len(new_articles):
            article = new_articles[idx]
            msg = format_message(article, r)
            send_telegram(msg)
            pushed += 1
            print(f"  推送: [{score}分] {article['title'][:50]}")

    print(f"共推送 {pushed} 条信号")

    # 6. 更新已读记录
    for a in new_articles:
        seen_ids.add(a["id"])
    save_seen_ids(seen_ids)

if __name__ == "__main__":
    main()
