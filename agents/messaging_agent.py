import os

import requests

from .agent import Agent
from .deals import Opportunity


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.CYAN

    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

    def __init__(self):
        self.log("Messaging Agent is initializing")
        self.do_push = os.getenv("DO_PUSH", "false").lower() == "true"
        self.do_text = os.getenv("DO_TEXT", "false").lower() == "true"
        self.log("Pushover enabled" if self.do_push else "Pushover disabled")
        self.log("Twilio enabled" if self.do_text else "Twilio disabled")

    def _format(self, opportunity: Opportunity) -> str:
        d = opportunity.deal
        return (
            f"Deal: {d.product_description}\n"
            f"Price: {d.price}\n"
            f"Estimate: {opportunity.estimate}\n"
            f"Discount: {opportunity.discount}\n"
            f"URL: {d.url}"
        )

    def _send_pushover(self, msg: str) -> None:
        token = os.getenv("PUSHOVER_TOKEN")
        user = os.getenv("PUSHOVER_USER")
        if not token or not user:
            self.log("Pushover credentials missing; skipping push")
            return

        try:
            response = requests.post(
                self.PUSHOVER_URL,
                data={"token": token, "user": user, "message": msg},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.log(f"Pushover transport error: {e!r}")
            return

        # Pushover returns HTTP 200 even on credential errors; the real status
        # is in the JSON body (1 = accepted, anything else = failure).
        try:
            payload = response.json()
        except ValueError:
            self.log("Pushover returned non-JSON response")
            return

        if payload.get("status") != 1:
            self.log(f"Pushover rejected message: {payload!r}")
        else:
            self.log("Pushover sent")

    def _send_twilio(self, msg: str) -> None:
        try:
            from twilio.base.exceptions import TwilioRestException
            from twilio.rest import Client
        except ImportError:
            self.log("twilio package not installed; skipping SMS")
            return

        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        from_ = os.getenv("TWILIO_FROM")
        to = os.getenv("MY_PHONE_NUMBER")
        if not all([sid, token, from_, to]):
            self.log("Twilio credentials missing; skipping SMS")
            return

        try:
            client = Client(sid, token)
            client.messages.create(body=msg, from_=from_, to=to)
            self.log("Twilio SMS sent")
        except TwilioRestException as e:
            self.log(f"Twilio API error: {e!r}")

    def alert(self, opportunity: Opportunity) -> None:
        msg = self._format(opportunity)
        if self.do_push:
            self._send_pushover(msg)
        if self.do_text:
            self._send_twilio(msg)
