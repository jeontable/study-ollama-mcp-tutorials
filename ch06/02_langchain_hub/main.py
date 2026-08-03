import bs4
from dotenv import load_dotenv
from langsmith import Client
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # .env 파일 로드

# 1. 뉴스 URL
news_url = """https://www.bbc.com/korean/articles/c166p510n79o"""

# 2. 뉴스 스크래핑
loader = WebBaseLoader(
    web_paths=([news_url]),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["css-bg8vrv", "css-1nude9v"]},
        )
    ),
)
news_array = loader.load()
news = news_array[0]

# 3. 요약에 사용할 프롬프트 불러오기
client = Client()

prompt = client.pull_prompt(
    "hellollama/news_summary",
    dangerously_pull_public_prompt=True,
)

# 4. Ollama 초기화
llm = ChatOllama(model="qwen3:8b", temperature=0)
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 5. 프롬프트를 실행할 체인생성
summary_chain = prompt | llm | StrOutputParser()

# 6. LLM에 질문
for chunk in summary_chain.stream({"news": news}):
    print(chunk, end="", flush=True)
