from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import ask_sdk_core.utils as ask_utils
import requests
import logging
import os
import re
import json
import openai
from dotenv import load_dotenv

## Retrieve environment variables
load_dotenv()
openai.api_key = OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MEALIE_API_KEY = os.getenv("MEALIE_API_KEY")
MEALIE_URL = os.getenv("MEALIE_URL")
MEALIE_LIST_ID = os.getenv("MEALIE_LIST_ID")

## Headers for authentication
MEALIE_HEADERS = {
    "Authorization": f"Bearer {MEALIE_API_KEY}",
    "Content-Type": "application/json"
}

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
        query = handler_input.request_envelope.request.intent.slots["query"].value
        
        ## Switch statement to handle different queries
        if query.startswith("add"):
            speak_output = add_to_mealie_list(query)
        else:
            # speak_output = "Please try again."
            session_attr = handler_input.attributes_manager.session_attributes
            if "chat_history" not in session_attr:
                session_attr["chat_history"] = []
            speak_output = generate_gpt_response(session_attr["chat_history"], query)
            session_attr["chat_history"].append((query, speak_output))

        return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask("Any other questions?")
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

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

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
        speak_output = "Leaving Chat G.P.T. mode"

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )

## https://github.com/k4l1sh/alexa-gpt
def generate_gpt_response(chat_history, new_question):
    """Generates a GPT response to a new question"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.openai.com/v1/chat/completions"
    messages = [{"role": "system", "content": "You are a helpful assistant. Answer in 50 words or less."}]
    for question, answer in chat_history[-10:]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": new_question})
    
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = response.json()
        if response.ok:
            return response_data['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response_data['error']['message']}"
    except Exception as e:
        return f"Error generating response: {str(e)}"

def add_to_mealie_list(query):
    try:
        ## Extract the item from the query
        item = re.sub(r"add\s+", "", query, flags=re.I).strip()

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

sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GeneralIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()