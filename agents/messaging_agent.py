import os
from agents.agent import Agent

class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.CYAN

    def __init__(self):
        self.log("Messaging Agent is initializing")
        self.do_push = os.getenv("DO_PUSH", "false").lower() == "true"
        self.do_text = os.getenv("DO_TEXT", "false").lower() == "true"
        self.log("Messaging Agent has initialized Pushover" if self.do_push else "Pushover disabled")
        self.log("Messaging Agent has initialized Twilio" if self.do_text else "Twilio disabled")

    def alert(self, opportunity):
        msg = f"Deal: {opportunity.deal.product_description}\nPrice: {opportunity.deal.price}\nEstimate: {opportunity.estimate}\nDiscount: {opportunity.discount}\nURL: {opportunity.deal.url}"
        if self.do_push:
            try:
                import requests
                token = os.getenv("PUSHOVER_TOKEN")
                user = os.getenv("PUSHOVER_USER")
                requests.post("https://api.pushover.net/1/messages.json", data={"token": token, "user": user, "message": msg}, timeout=10)
                self.log("Pushover sent")
            except Exception as e:
                self.log(f"Pushover error: {e}")
        if self.do_text:
            try:
                from twilio.rest import Client
                client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
                client.messages.create(body=msg, from_=os.getenv("TWILIO_FROM"), to=os.getenv("MY_PHONE_NUMBER"))
                self.log("Twilio SMS sent")
            except Exception as e:
                self.log(f"Twilio error: {e}")
