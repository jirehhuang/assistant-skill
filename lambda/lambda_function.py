from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_core.skill_builder import CustomSkillBuilder
from ask_sdk_dynamodb.adapter import DynamoDbAdapter
import ask_sdk_core.utils as ask_utils
import pytz
import logging
import boto3
from datetime import datetime
from collections import defaultdict

##====================================================================================================##

import requests
import os
import re
import json
import openai
from dotenv import load_dotenv
from collections import defaultdict

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
NOTION_TASK_ID_CHATS = os.getenv("NOTION_TASK_ID_CHATS")

S3_URI_1MIN_SILENCE = os.getenv("S3_URI_1MIN_SILENCE")

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
MODEL_MAIN = "gpt-4o-mini"
# MODEL_MAIN = "google/gemini-2.0-flash-thinking-exp:free"
MODEL_FREE = "gpt-4o-mini"
# MODEL_FREE = "google/gemini-2.0-flash-thinking-exp:free"
MODEL_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_HEADERS = OPENROUTER_HEADERS

## Parameters
MAX_INPUT_CHARS = 10000
MAX_OUTPUT_TOKENS = 300

##====================================================================================================##

## Initialize the DynamoDB adapter
ddb_region = os.environ.get('DYNAMODB_PERSISTENCE_REGION')
ddb_table_name = os.environ.get('DYNAMODB_PERSISTENCE_TABLE_NAME')

ddb_resource = boto3.resource('dynamodb', region_name=ddb_region)
dynamodb_adapter = DynamoDbAdapter(table_name=ddb_table_name, create_table=False, dynamodb_resource=ddb_resource)

TZ = pytz.timezone('America/Los_Angeles')

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
        
        session_attr = check_session_attr(session_attr = handler_input.attributes_manager.session_attributes, persistent_attr = handler_input.attributes_manager.persistent_attributes)

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
            if query in ["no", "stop", "exit", "quit", "cancel", "close", "nevermind", "no thank you", "save and exit"]:
                return SessionEndedRequestHandler().handle(handler_input)
            
            session_attr = check_session_attr(session_attr = handler_input.attributes_manager.session_attributes, persistent_attr = handler_input.attributes_manager.persistent_attributes)

            ## Switch statement to handle different queries
            if query in ["wait", "hold on", "pause", "hang on", "just a moment", "let me think", "give me a second"]:
                ## Wait for x seconds
                speak_output = f"<speak>Sure thing. <audio src='{S3_URI_1MIN_SILENCE}'/></speak>"
            elif query.startswith("add"):
                ## Add to list
                item = re.sub(r"^add\s+", "", query, flags=re.I).strip()
                speak_output = smart_add_item(item, session_attr)
            elif query.startswith("save"):
                ## Attempt to save output
                speak_output = "Saved"
                session_page_id = save_chat_history_to_notion(page_id = session_attr.get("session_page_id", None), chat_history=session_attr.get("chat_history", []))
                if " " not in session_page_id:
                    session_attr["session_page_id"] = session_page_id
                else:
                    speak_output = session_page_id  # Error message
            elif ((any(keyword in query.lower() for keyword in ["get", "load", "retrieve"]) and 
                   any(phrase in query.lower() for phrase in ["shopping list", "mealie list"])) or 
                  any(phrase in query.lower() for phrase in ["my shopping list", "my mealie list"])):
                ## Get Mealie shopping list
                speak_output = get_shopping_list()
            else:
                ## General LLM response
                speak_output = general_response(query, session_attr)
            
            timestamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
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
        speak_output = "OK"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .set_should_end_session(True)
                .response
        )

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        ## Log the reason for session end
        reason = handler_input.request_envelope.request.reason
        logger.info(f"Session ended with reason: {reason}")
        
        ## Attempt to save output
        session_attr = handler_input.attributes_manager.session_attributes
        handler_input.attributes_manager.persistent_attributes = session_attr
        handler_input.attributes_manager.save_persistent_attributes()
        save_chat_history_to_notion(page_id = session_attr.get("session_page_id", None), chat_history=session_attr.get("chat_history", []))

        return handler_input.response_builder.response

def check_session_attr(session_attr = {}, persistent_attr = {}):
    ## Attempt to copy from persistent attributes
    if not session_attr:
        session_attr.update(persistent_attr)
        session_attr.pop("launch_timestamp", None)
    
    ## Initialize launch timestamp
    if "launch_timestamp" not in session_attr:
        session_attr["launch_timestamp"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    
    if "chat_history" not in session_attr:
        session_attr["chat_history"] = []
    ## TODO: Truncate
    
    ## Default models
    if "model_main" not in session_attr:
        session_attr["model_main"] = MODEL_MAIN
    if "model_free" not in session_attr:
        session_attr["model_free"] = MODEL_FREE
    session_attr["model_api_url"] = MODEL_API_URL
    session_attr["model_headers"] = MODEL_HEADERS
    
    ## Default Parameters
    if "max_input_chars" not in session_attr:
        session_attr["max_input_chars"] = MAX_INPUT_CHARS
    if "max_output_tokens" not in session_attr:
        session_attr["max_output_tokens"] = MAX_OUTPUT_TOKENS
    
    return session_attr

def general_response(query, session_attr = {}):
    """Generates a general response to a query."""
    try:
        ## Check session attributes
        session_attr = check_session_attr(session_attr)

        messages = [{"role": "system", "content": 
                     "You are an AI voice assistant designed to engage in natural, real-time conversations. "
                     "Assist users by answering questions, providing information, asking clarifying questions, offering suggestions, and acting as a sounding board. "
                     "Be conversationally concise with your responses. "
                     "Avoid unnecessary elaboration unless the user requests more details. "
                     "If you do not know the answer to a question, admit it honestly and suggest alternative ways the user might find the information."}]
        
        total_chars = 0
        for timestamp, question, answer in reversed(session_attr["chat_history"]):
            user_message = f"[{timestamp}] {question}"
            assistant_message = answer
            if (total_chars + len(user_message) + len(assistant_message)) > session_attr["max_input_chars"]:
                break
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})
            total_chars += len(user_message) + len(assistant_message)
        
        messages.reverse()
        messages.append({"role": "user", "content": query})
        
        data = {
            "model": session_attr["model_free"],
            "messages": messages,
            "max_tokens": session_attr["max_output_tokens"],
            "temperature": 0
        }
        
        response = requests.post(session_attr["model_api_url"], 
                                 headers=session_attr["model_headers"], data=json.dumps(data))
        response_data = response.json()
        if response.ok:
            return response_data['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response_data.get('error', {}).get('message', 'Unknown error')}"
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

def create_notion_task(title, parent_page_id=NOTION_TASK_ID_GENERAL):
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
                            "id": parent_page_id
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
            page_id = response.json().get("id")
            print(f"Page '{title}' created successfully with ID: {page_id}.")
            return page_id
        else:
            print(f"Failed to create page '{title}': {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

def find_or_create_notion_task(title, parent_page_id=NOTION_TASK_ID_GENERAL):
    try:
        # Search for a page with the same title and parent task
        query_payload = {
            "filter": {
                "and": [
                    {
                        "property": "Name",
                        "title": {
                            "equals": title
                        }
                    },
                    {
                        "property": "Parent task",
                        "relation": {
                            "contains": parent_page_id
                        }
                    }
                ]
            }
        }

        # Send the request to search for the page
        search_response = requests.post(f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID_TASKS}/query",
                                        headers=NOTION_HEADERS,
                                        data=json.dumps(query_payload))

        if search_response.status_code == 200:
            results = search_response.json().get("results", [])
            if results:
                page_id = results[0]["id"]
                print(f"Page '{title}' already exists with ID: {page_id}.")
                return page_id
            else:
                print(f"No existing page found for title '{title}'. Creating a new one.")
        else:
            print(f"Failed to search for page '{title}': {search_response.status_code}, {search_response.text}")
            return None

        # Create the page if it doesn't exist
        return create_notion_task(title, parent_page_id)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

def smart_add_item(input_value, session_attr = {}):
    try:
        ## Check session attributes
        session_attr = check_session_attr(session_attr)

        ## Define the prompt
        prompt = f"""
        You are a smart assistant. Analyze the following input and split it into individual items. 
        For each item, determine if it is a shopping item or a task/note. 
        Respond with a JSON array where each element is a dictionary with "function" and "item" attributes.
        The output format should be a raw JSON array without formatting (such as code block) that can be interpretable with json.loads().
        "function" should be "add_to_mealie_list" for shopping items and "create_notion_task" for tasks and notes.
        Input: "{input_value}"
        """

        ## Call model to analyze the input
        payload = {
            "model": session_attr["model_main"],
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(session_attr["model_api_url"], 
                                 headers=session_attr["model_headers"], data=json.dumps(payload))

        if response.status_code != 200:
            return f"Error calling {payload['model']}: {response.status_code}, {response.text}"

        ## Parse the response
        response_json = response.json()
        response_json = response_json["choices"][0]["message"]["content"]
        decisions = json.loads(response_json)
        print(decisions)

        mealie_items = []
        notion_items = []

        ## Execute the appropriate function for each decision
        for decision in decisions:
            if decision["function"] == "add_to_mealie_list":
                add_to_mealie_list(decision["item"])
                mealie_items.append(decision["item"])
            elif decision["function"] == "create_notion_task":
                create_notion_task(decision["item"])  # Create regardless of existing task
                notion_items.append(decision["item"])
            print(f"Decision: {decision['function']} - {decision['item']}")

        ## Construct the result string
        mealie_str = f"added {', '.join(mealie_items)} to mealie" if mealie_items else ""
        notion_str = f"added {', '.join(notion_items)} to notion" if notion_items else ""
        result_str = " and ".join(filter(None, [mealie_str, notion_str]))

        return result_str

    except Exception as e:
        return f"Error occurred: {str(e)}"

def save_chat_history_to_notion(page_id=None, chat_history=[]):
    """
    Save chat history to a Notion page.
    """
    try:
        if not chat_history:
            return "No chat history provided to save."
        if page_id is None:
            page_id = find_or_create_notion_task(f"Session {chat_history[0][0]}", parent_page_id=NOTION_TASK_ID_CHATS)

        ## Fetch existing content from the Notion page
        response = requests.get(f"https://api.notion.com/v1/blocks/{page_id}/children", headers=NOTION_HEADERS)
        if response.status_code != 200:
            return f"Failed to fetch page content: {response.status_code}, {response.text}"
        
        existing_content = response.json()
        existing_texts = set()
        for block in existing_content.get("results", []):
            if block["type"] == "paragraph" and block["paragraph"]["rich_text"]:
                existing_texts.add(block["paragraph"]["rich_text"][0]["text"]["content"])
        
        ## Prepare new chat history to add
        new_blocks = []
        for timestamp, query, speak_output in chat_history:
            try:
                # Create main block for the timestamp
                if timestamp not in existing_texts:  # Avoid duplicates
                    new_blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": timestamp}}],
                            "children": [
                                {
                                    "object": "block",
                                    "type": "paragraph",
                                    "paragraph": {
                                        "rich_text": [
                                            {
                                                "type": "text",
                                                "text": {
                                                "content": query
                                                },
                                                "annotations": {
                                                "bold": True,
                                                "italic": True
                                                }
                                            }
                                        ]
                                    }
                                },
                                {
                                    "object": "block",
                                    "type": "paragraph",
                                    "paragraph": {
                                        "rich_text": [{"type": "text", "text": {"content": f"{speak_output}"}}]
                                    }
                                }
                            ]
                        }
                    })
            except Exception as e:
                return f"Error processing chat history entry: {str(e)}"
        
        if not new_blocks:
            return f"{page_id}"
        
        ## Add new chat history to the Notion page
        payload = {"children": new_blocks}
        response = requests.patch(f"https://api.notion.com/v1/blocks/{page_id}/children", 
                                  headers=NOTION_HEADERS, data=json.dumps(payload))
        
        if response.status_code == 200:
            return f"{page_id}"
        else:
            return f"Error updating page: {response.status_code}, {response.text}"
    
    except Exception as e:
        return f"Error occurred: {str(e)}"

def get_shopping_list():
    ## Retrieve items in shopping list
    response = requests.get(f"{MEALIE_URL}/households/shopping/items?orderBy=checked&orderDirection=asc&page=1&perPage=100&checked=false", 
                        headers=MEALIE_HEADERS)

    items = [{"item": item["display"], "label": item["label"]["name"] if "label" in item and item["label"] else "No Label"} 
            for item in response.json()["items"] if item["checked"] == False]

    ## Group items by their labels
    grouped_items = defaultdict(list)
    for entry in items:
        grouped_items[entry['label']].append(entry['item'])

    ## Format the string
    total_items = sum(len(items) for items in grouped_items.values())
    items_string = f"{total_items} items. " + ". ".join(
        f"{len(items)} {label}: {', '.join(items)}" for label, items in grouped_items.items()
    )
    return items_string


sb = CustomSkillBuilder(persistence_adapter = dynamodb_adapter)

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GeneralIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()