import logging
from core.orchestrator import BaseAgent
from core.router import BuddyRouter
from core.tools import ToolRegistry

logger = logging.getLogger(__name__)

class FinanceWorkflow:
    def __init__(self, router: BuddyRouter, tool_registry: ToolRegistry):
        self.router = router
        self.tool_registry = tool_registry
        
        # Step 2A: The Scraper (News Gatherer)
        self.finance_scraper = BaseAgent("finance_scraper", router, tool_registry)
        
        # Step 2B: The Processor (Macro Analyst)
        self.finance_processor = BaseAgent("finance_processor", router, tool_registry)
        
        # Step 2C: The Matcher (Quant Data Integrator)
        self.finance_matcher = BaseAgent("finance_matcher", router, tool_registry)
        
        # Step 2D: The Validator (Risk Manager & Editor)
        self.finance_validator = BaseAgent("finance_validator", router, tool_registry)
        
    async def run(self, user_query: str, model_id: str = "gemini/gemini-2.5-flash") -> dict:
        """Full pipeline execution from Scraper to Validator."""
        logger.info(f"Running full finance pipeline for: {user_query} using model {model_id}")
        
        # 1. Run Scraper
        scraper_prompt = (
            "You are a financial news scraper. Your ONLY job is to take the user's query "
            "and use the web_search tool to find the most recent, accurate macroeconomic "
            "and sector-specific news today. Output ONLY the factual news you found."
        )
        messages = [
            {"role": "system", "content": scraper_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Async generators must be iterated over
        async for item in self.finance_scraper.run(messages, model_id):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                result = item
        news_output = result[0]
        
        # 2. Run Processor
        processor_prompt = (
            "You are a Macroeconomic Sentiment Analyst. You will receive raw news data. "
            "Extract the core sentiment (Bullish/Bearish/Neutral) and summarize the "
            "macroeconomic impact concisely. No tools required."
        )
        proc_messages = [
            {"role": "system", "content": processor_prompt},
            {"role": "user", "content": f"User Subject: {user_query}\n\nRaw News Data:\n{news_output}"}
        ]
        
        async for item in self.finance_processor.run(proc_messages, model_id):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                proc_result = item
        sentiment_output = proc_result[0]
        
        # 3. Run Matcher
        matcher_prompt = (
            "You are a Quantitative Data Integrator. You have access to the get_live_stock_info tool. "
            "Your job is to identify relevant stock tickers from the user's query or the macro sentiment, "
            "fetch their live financial data, and combine it with the macroeconomic sentiment provided. "
            "Only output the hard data and the contextual matching. Do not format as a final report yet."
        )
        matcher_messages = [
            {"role": "system", "content": matcher_prompt},
            {"role": "user", "content": f"User Query: {user_query}\n\nMacro Sentiment:\n{sentiment_output}"}
        ]
        
        async for item in self.finance_matcher.run(matcher_messages, model_id):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                match_result = item
        quant_output = match_result[0]
        
        # 4. Run Validator
        validator_prompt = (
            "You are a strict Financial Compliance Officer and Editor. Your job is to take the "
            "raw quantitative and sentiment data, ensure there are no contradictory claims, "
            "and format it into a clear, institutional-grade Markdown report.\n"
            "CRITICAL: You MUST append a strict disclaimer at the end stating that this is not "
            "financial advice and is for informational purposes only. Do not use any tools."
        )
        val_messages = [
            {"role": "system", "content": validator_prompt},
            {"role": "user", "content": f"Quant & Sentiment Data:\n{quant_output}"}
        ]
        
        async for item in self.finance_validator.run(val_messages, model_id):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                val_result = item
        final_report = val_result[0]
        
        yield {
            "type": "result",
            "news": news_output,
            "sentiment": sentiment_output,
            "quant_data": quant_output,
            "final_report": final_report
        }

    async def run_matcher_test(self, user_query: str):
        """Isolated test method for Step 2A -> 2B -> 2C integration"""
        logger.info(f"Running scraper -> processor -> matcher test for: {user_query}")
        
        # 1. Run Scraper
        scraper_prompt = (
            "You are a financial news scraper. Your ONLY job is to take the user's query "
            "and use the web_search tool to find the most recent, accurate macroeconomic "
            "and sector-specific news today. Output ONLY the factual news you found."
        )
        messages = [
            {"role": "system", "content": scraper_prompt},
            {"role": "user", "content": user_query}
        ]
        
        async for item in self.finance_scraper.run(messages, "gemini/gemini-2.5-flash"):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                news_output = item[0]
        
        # 2. Run Processor
        processor_prompt = (
            "You are a Macroeconomic Sentiment Analyst. You will receive raw news data. "
            "Extract the core sentiment (Bullish/Bearish/Neutral) and summarize the "
            "macroeconomic impact concisely. No tools required."
        )
        proc_messages = [
            {"role": "system", "content": processor_prompt},
            {"role": "user", "content": f"User Subject: {user_query}\n\nRaw News Data:\n{news_output}"}
        ]
        
        async for item in self.finance_processor.run(proc_messages, "gemini/gemini-2.5-flash"):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                sentiment_output = item[0]
        
        # 3. Run Matcher
        matcher_prompt = (
            "You are a Quantitative Data Integrator. You have access to the get_live_stock_info tool. "
            "Your job is to identify relevant stock tickers from the user's query or the macro sentiment, "
            "fetch their live financial data, and combine it with the macroeconomic sentiment provided. "
            "Only output the hard data and the contextual matching. Do not format as a final report yet."
        )
        matcher_messages = [
            {"role": "system", "content": matcher_prompt},
            {"role": "user", "content": f"User Query: {user_query}\n\nMacro Sentiment:\n{sentiment_output}"}
        ]
        
        async for item in self.finance_matcher.run(matcher_messages, "gemini/gemini-2.5-flash"):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                quant_output = item[0]
        
        yield {
            "type": "result",
            "news": news_output,
            "sentiment": sentiment_output,
            "quant_data": quant_output
        }

    async def run_processor_test(self, user_query: str):
        """Isolated test method for Step 2A -> Step 2B integration"""
        logger.info(f"Running scraper -> processor test for: {user_query}")
        
        # 1. Run Scraper
        scraper_prompt = (
            "You are a financial news scraper. Your ONLY job is to take the user's query "
            "and use the web_search tool to find the most recent, accurate macroeconomic "
            "and sector-specific news today. Output ONLY the factual news you found."
        )
        messages = [
            {"role": "system", "content": scraper_prompt},
            {"role": "user", "content": user_query}
        ]
        
        async for item in self.finance_scraper.run(messages, "gemini/gemini-2.5-flash"):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                news_output = item[0]
        
        # 2. Run Processor
        processor_prompt = (
            "You are a Macroeconomic Sentiment Analyst. You will receive raw news data. "
            "Extract the core sentiment (Bullish/Bearish/Neutral) and summarize the "
            "macroeconomic impact concisely. No tools required."
        )
        proc_messages = [
            {"role": "system", "content": processor_prompt},
            {"role": "user", "content": f"User Subject: {user_query}\n\nRaw News Data:\n{news_output}"}
        ]
        
        async for item in self.finance_processor.run(proc_messages, "gemini/gemini-2.5-flash"):
            if isinstance(item, dict) and item.get("type") == "status":
                yield item
            else:
                sentiment_output = item[0]
        
        yield {
            "type": "result",
            "news": news_output,
            "sentiment": sentiment_output
        }

    async def run_scraper_test(self, user_query: str) -> str:
        """Isolated test method for Step 2A (Scraper)"""
        logger.info(f"Running scraper test for: {user_query}")
        
        scraper_prompt = (
            "You are a financial news scraper. Your ONLY job is to take the user's query "
            "and use the web_search tool to find the most recent, accurate macroeconomic "
            "and sector-specific news today. Output ONLY the factual news you found."
        )
        
        messages = [
            {"role": "system", "content": scraper_prompt},
            {"role": "user", "content": user_query}
        ]
        
        async for item in self.finance_scraper.run(messages, "gemini/gemini-2.5-flash"):
            if not isinstance(item, dict):
                result = item
        news_output = result[0]
        return news_output
