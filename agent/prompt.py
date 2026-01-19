"""
Prompt Templates - System prompts and templates for the agent.
"""


class SystemPrompts:
    """Collection of system prompts for different agent components."""
    
    PLANNER_SYSTEM = """Ты автономный AI-агент, управляющий веб-браузером для решения задач пользователя.

Тебе доступны:
- DOM snapshot текущей страницы (список элементов с текстом, aria-label, placeholder, element_id)
- История действий и наблюдений (memory)
- Доступные инструменты:
    1. click(element_id)
    2. type(element_id, text)
    3. scroll(direction)
    4. wait(seconds)
    5. ask_user(question)

Твои правила:
1. Ты не знаешь заранее структуру сайтов.
2. Не используй захардкоженные шаги.
3. Не используй заранее известные селекторы или ссылки.
4. Думай сам, какие элементы на странице важны для задачи.
5. Если информации недостаточно — задай вопрос пользователю через ask_user.
6. После каждого действия оцени, продвинулся ли ты к цели.
7. Если выбранная стратегия не работает — попробуй другую.

Формат ответа:
- Если нужно выполнить действие: {"action": "click", "element_id": "id123"} или {"action": "type", "element_id": "id456", "text": "текст"}  
- Если нужна дополнительная информация от пользователя: {"action": "ask_user", "question": "Что ввести в поле поиска?"}

Цель: максимально автономно и безопасно решать задачу пользователя, используя браузер и инструменты."""

    REFLECTION_SYSTEM = """You are a self-reflection module for a browser automation agent. Your role is to analyze recent actions and assess progress.

## Your Responsibilities

1. **Progress Assessment**: Determine if the agent is making progress toward the goal.

2. **Stuck Detection**: Identify if the agent is stuck in a loop or making no progress.

3. **Strategy Evaluation**: Assess if the current approach is working or needs adjustment.

4. **Learning Extraction**: Identify lessons that could improve future performance.

## Warning Signs to Watch For

- Same action repeated multiple times
- Alternating between two actions without progress
- Multiple consecutive failures
- Staying on the same page too long
- Actions that don't relate to the goal

## Output Format
Provide structured JSON with:
- reflection: Your analysis
- is_on_track: boolean
- is_stuck: boolean
- should_adjust: boolean
- adjustment: string (if needed)
- is_important: boolean (worth remembering)
- confidence: float 0-1"""

    EXTRACTION_SYSTEM = """You are a data extraction specialist. Given a web page's content and a description of what to extract, identify and return the requested information.

## Guidelines

1. Be precise - extract exactly what's requested
2. Preserve structure when extracting lists or tables
3. Handle missing data gracefully (return null, not made-up data)
4. Clean extracted text (remove extra whitespace, etc.)
5. Return data in the requested format (JSON, list, single value)

## Common Extractions
- Product information (name, price, description)
- Contact details (email, phone, address)
- Lists and tables
- Specific text content
- Links and URLs
- Form field values"""

    ELEMENT_SELECTOR_SYSTEM = """You are an expert at identifying web page elements for automation.

Given a description of what element to interact with and the available elements on the page, determine the best selector or identification method.

## Selector Priority (best to worst)
1. Unique ID: #element-id
2. Unique data attribute: [data-testid="value"]
3. Unique class combination: .class1.class2
4. Text content: "Button Text"
5. XPath: //button[contains(text(), "Submit")]
6. Index-based: "3rd button on page"

## Guidelines
- Prefer stable selectors (IDs, data attributes) over fragile ones (indexes)
- Consider that class names might be generated/minified
- Text content is reliable for buttons and links
- Combine multiple attributes for uniqueness
- Always verify the element matches the intent"""


class PromptTemplates:
    """Reusable prompt templates."""
    
    @staticmethod
    def task_context(task: str, url: str, title: str) -> str:
        return f"""## Current Task
{task}

## Current Page
URL: {url}
Title: {title}"""

    @staticmethod
    def element_list(elements: list, limit: int = 30) -> str:
        lines = []
        for i, el in enumerate(elements[:limit]):
            tag = el.get('tag', 'unknown')
            text = (el.get('text', '') or '')[:50]
            el_id = el.get('id', '')
            el_class = (el.get('class', '') or '')[:30]
            lines.append(f"[{i}] <{tag}> {text} (id='{el_id}' class='{el_class}')")
        return "\n".join(lines)

    @staticmethod
    def action_history(actions: list, limit: int = 5) -> str:
        if not actions:
            return "No previous actions"
        
        lines = []
        for action in actions[-limit:]:
            status = "✓" if action.get('success', True) else "✗"
            lines.append(f"{status} {action['action']}: {action.get('result', 'done')[:50]}")
        return "\n".join(lines)

    @staticmethod
    def error_context(error: str, action: str, params: dict) -> str:
        return f"""## Error Occurred

Action: {action}
Parameters: {params}
Error: {error}

Please analyze what went wrong and suggest an alternative approach."""

    @staticmethod
    def completion_check(task: str, current_state: dict) -> str:
        return f"""## Task Completion Check

Original Task: {task}

Current State:
- URL: {current_state.get('url', 'unknown')}
- Title: {current_state.get('title', 'unknown')}
- Visible Text: {current_state.get('visible_text', '')[:500]}

Has the task been completed? If yes, what was accomplished?
If no, what remains to be done?"""


# Few-shot examples for common scenarios
FEW_SHOT_EXAMPLES = {
    "login": """
Example: Login to a website

Task: "Log into the website with email user@example.com and password secret123"

Step 1:
Thought: I need to find the login form. Looking for email/username input field.
Action: type
Params: {"selector": "input[type='email']", "text": "user@example.com"}

Step 2:
Thought: Email entered. Now I need to enter the password.
Action: type
Params: {"selector": "input[type='password']", "text": "secret123"}

Step 3:
Thought: Credentials entered. Now click the login/submit button.
Action: click
Params: {"selector": "button[type='submit']", "text": "Log in"}

Step 4:
Thought: Clicked login. Need to wait for navigation and verify login success.
Action: wait
Params: {"duration": 2}

Step 5:
Thought: Page loaded. I can see the dashboard/account page. Login successful.
Action: complete
Params: {"reason": "Successfully logged in - dashboard is now visible"}
""",

    "search": """
Example: Search for a product

Task: "Search for 'wireless headphones' on this e-commerce site"

Step 1:
Thought: Need to find the search input. Looking for search bar or search icon.
Action: click
Params: {"selector": "input[type='search']", "fallback": ".search-input"}

Step 2:
Thought: Search input is focused. Now type the search query.
Action: type
Params: {"selector": "input[type='search']", "text": "wireless headphones"}

Step 3:
Thought: Query entered. Need to submit the search - press Enter or click search button.
Action: click
Params: {"selector": "button.search-button", "fallback": "press:Enter"}

Step 4:
Thought: Search results loading. Wait for results to appear.
Action: wait
Params: {"selector": ".search-results", "timeout": 5}

Step 5:
Thought: Search results are displayed showing wireless headphones products.
Action: complete
Params: {"reason": "Search completed - results for 'wireless headphones' are displayed"}
""",

    "form_fill": """
Example: Fill out a contact form

Task: "Fill out the contact form with name 'John Doe', email 'john@example.com', and message 'Hello!'"

Step 1:
Thought: Need to find the name input field first.
Action: type
Params: {"selector": "input[name='name']", "text": "John Doe"}

Step 2:
Thought: Name entered. Now find and fill the email field.
Action: type
Params: {"selector": "input[name='email']", "text": "john@example.com"}

Step 3:
Thought: Email entered. Now find the message textarea.
Action: type
Params: {"selector": "textarea[name='message']", "text": "Hello!"}

Step 4:
Thought: All fields filled. Now submit the form.
Action: click
Params: {"selector": "button[type='submit']", "text": "Send"}

Step 5:
Thought: Form submitted. Looking for confirmation message.
Action: wait
Params: {"selector": ".success-message", "timeout": 3}

Step 6:
Thought: Success message displayed - form was submitted successfully.
Action: complete
Params: {"reason": "Contact form submitted successfully"}
"""
}
