"""
AI Browser Agent - Точка входа
Автономный агент для управления браузером с использованием LLM.
"""

import asyncio
import argparse
import logging
import os
from dotenv import load_dotenv
from agent.agent import BrowserAgent
from browser.playwright import BrowserController

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='AI Browser Agent')
    parser.add_argument('--task', type=str, required=True, help='Task description for the agent')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    parser.add_argument('--max-steps', type=int, default=50, help='Maximum steps before stopping')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='LLM model to use (default: gpt-4o-mini, also supports: gpt-3.5-turbo, gpt-4, gpt-4-turbo)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    logger.info(f"Starting AI Browser Agent")
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max steps: {args.max_steps}")
    
    browser = BrowserController(headless=args.headless)
    agent = BrowserAgent(
        browser=browser,
        model=args.model,
        max_steps=args.max_steps
    )
    
    try:
        await browser.start()
        result = await agent.run(args.task)
        logger.info(f"Task completed: {result}")
    except KeyboardInterrupt:
        logger.info("Agent interrupted by user")
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
    finally:
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
