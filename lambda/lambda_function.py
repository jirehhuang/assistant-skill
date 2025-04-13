from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import ask_sdk_core.utils as ask_utils
import logging
from datetime import datetime

import requests
import os
import re
import json
import openai
from dotenv import load_dotenv


## Retrieve environment variables
load_dotenv()

openai.api_key = OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MEALIE_API_KEY = os.getenv("MEALIE_API_KEY")
MEALIE_URL = os.getenv("MEALIE_URL")
MEALIE_LIST_ID = os.getenv("MEALIE_LIST_ID")

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID_TASKS = os.getenv("NOTION_DATABASE_ID_TASKS")
NOTION_TASK_ID_GENERAL = os.getenv("NOTION_TASK_ID_GENERAL")

## Headers for authentication
OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}
MEALIE_HEADERS = {
    "Authorization": f"Bearer {MEALIE_API_KEY}",
    "Content-Type": "application/json"
}
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

## API settings for selected model
# MODEL = "gpt-4o-mini"
# MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"
MODEL = "google/gemini-2.0-flash-thinking-exp:free"

if MODEL == "gpt-4o-mini":
    MODEL_API_URL = "https://api.openrouter.ai/v1/generate"
    MODEL_HEADERS = OPENAI_HEADERS
else:
    MODEL_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL_HEADERS = OPENROUTER_HEADERS

## Parameters
MAX_INPUT_CHARS = 10000
MAX_OUTPUT_TOKENS = 300

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "How can I help?"

        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

class GeneralIntentHandler(AbstractRequestHandler):
    """Handler for General Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("GeneralIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        try:
            query = handler_input.request_envelope.request.intent.slots["query"].value

            ## Exit skill
            if query in ["no", "stop", "exit", "quit", "cancel", "close", "nevermind", "no thank you"]:
                return CancelOrStopIntentHandler().handle(handler_input)
            
            session_attr = handler_input.attributes_manager.session_attributes
            if "chat_history" not in session_attr:
                session_attr["chat_history"] = []

            ## Switch statement to handle different queries
            if query in ["wait", "hold on", "pause", "hang on", "just a moment", "let me think", "give me a second"]:
                ## TODO: Not working
                speak_output = "<speak>Sure, I'll give you a minute. <audio src='https://github.com/anars/blank-audio/raw/refs/heads/master/10-seconds-of-silence.mp3'/></speak>"
            elif query.startswith("add"):
                item = re.sub(r"^add\s+", "", query, flags=re.I).strip()
                speak_output = smart_add_item(item)
            else:
                speak_output = general_response(session_attr["chat_history"], query)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session_attr["chat_history"].append((timestamp, query, speak_output))

            return (
                    handler_input.response_builder
                        .speak(speak_output)
                        .ask("Anything else?")
                        .response
                )
        except Exception as e:
            speak_output = f"Error handling GeneralIntent: {str(e)}"
            logger.error(speak_output, exc_info=True)
            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("Would you like to try again?")
                    .response
            )

class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors."""
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Please try again."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Take care"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )

def general_response(chat_history, new_question):
    """Generates a general response to a new question."""
    messages = [{"role": "system", "content": 
                 "You are a helpful assistant named Jarvis. "
                 "Answer as succinctly as possible while maintaining clarity and completeness. "
                 "If you’re having trouble simplifying your response, provide a brief summary and prompt the user for desired details."}]
    
    total_chars = 0
    for timestamp, question, answer in reversed(chat_history):
        user_message = f"[{timestamp}]: {question}"
        assistant_message = answer
        if (total_chars + len(user_message) + len(assistant_message)) > MAX_INPUT_CHARS:
            break
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
        total_chars += len(user_message) + len(assistant_message)
    
    messages.append({"role": "user", "content": new_question})
    
    data = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0
    }
    try:
        response = requests.post(MODEL_API_URL, headers=MODEL_HEADERS, data=json.dumps(data))
        response_data = response.json()
        if response.ok:
            return response_data['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response_data['error']['message']}"
    except Exception as e:
        return f"Error generating response: {str(e)}"

def add_to_mealie_list(item):
    try:
        ## Create the payload for the API request
        payload = {
            "shoppingListId": MEALIE_LIST_ID,
            "note": item
        }

        ## Send the request to add the item to the shopping list
        response = requests.post(f"{MEALIE_URL}/households/shopping/items", 
                                 headers=MEALIE_HEADERS, data=json.dumps(payload))

        if response.status_code == 201:
            return f"Successfully added {item}"
        else:
            return f"Unsuccessful adding {item}: {response.status_code}, {response.text}"
        
    except Exception as e:
        return f"Error occurred: {str(e)}"

def create_notion_task(title, parent_task_id=NOTION_TASK_ID_GENERAL):
    try:
        # Create the payload for the API request
        payload = {
            "parent": {"database_id": NOTION_DATABASE_ID_TASKS},
            "properties": {
                "Name": {
                    "title": [
                        {
                            "text": {
                                "content": title
                            }
                        }
                    ]
                },
                "Parent task": {
                    "relation": [
                        {
                            "id": parent_task_id
                        }
                    ]
                }
            }
        }

        # Send the request to create the page
        response = requests.post("https://api.notion.com/v1/pages", 
                                 headers=NOTION_HEADERS, 
                                 data=json.dumps(payload))

        if response.status_code == 200:
            print(f"Page '{title}' created successfully.")
        else:
            print(f"Failed to create page '{title}': {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def smart_add_item(input_value):
    try:
        ## Define the prompt
        prompt = f"""
        You are a smart assistant. Analyze the following input and split it into individual items or tasks. 
        For each item or task, determine if it is a shopping item or a task. 
        Respond with a JSON array where each element is a dictionary with "function" and "item" attributes.
        The output format should be a raw JSON array without formatting (such as code block) that can be interpretable with json.loads().
        "function" should be "add_to_mealie_list" for shopping items and "create_notion_task" for tasks.
        Input: "{input_value}"
        """

        ## Call model to analyze the input
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(MODEL_API_URL, 
                                 headers=MODEL_HEADERS, data=json.dumps(payload))

        if response.status_code != 200:
            return f"Error calling {payload['model']}: {response.status_code}, {response.text}"

        ## Parse the response
        gpt_response = response.json()
        gpt_content = gpt_response["choices"][0]["message"]["content"]
        decisions = json.loads(gpt_content)
        print(decisions)

        mealie_items = []
        notion_items = []

        ## Execute the appropriate function for each decision
        for decision in decisions:
            if decision["function"] == "add_to_mealie_list":
                add_to_mealie_list(decision["item"])
                mealie_items.append(decision["item"])
            elif decision["function"] == "create_notion_task":
                create_notion_task(decision["item"])
                notion_items.append(decision["item"])
            print(f"Decision: {decision['function']} - {decision['item']}")

        ## Construct the result string
        mealie_str = f"added {', '.join(mealie_items)} to mealie" if mealie_items else ""
        notion_str = f"added {', '.join(notion_items)} to notion" if notion_items else ""
        result_str = " and ".join(filter(None, [mealie_str, notion_str]))

        return result_str

    except Exception as e:
        return f"Error occurred: {str(e)}"

sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GeneralIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()