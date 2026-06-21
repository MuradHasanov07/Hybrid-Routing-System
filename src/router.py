"""
router.py

The "hybrid routing" half of the system:

    classifier label == "Spam"        -> TemplateResponder (cheap, instant, no LLM call)
    classifier label != "Spam"        -> GPTRouter         (expensive, detailed response)

This is the cost-saving mechanism described in the assignment: only valid/complex
messages reach the (simulated) GPT model; spam is filtered out and answered with a
free template instead.

GPTRouter.generate_response() is a MOCK - it does not call a real LLM API. It is
written with the same function signature a real call would use, so swapping in a
real API (e.g. OpenAI's `openai.chat.completions.create(...)`) later only requires
replacing the body of `_call_llm_api`, not anything else in this file or in
pipeline.py. This is documented in the README.
"""

import random
import time


class TemplateResponder:
    """Handles Spam messages with a fixed, free, instant template reply -
    no LLM call needed."""

    SPAM_REPLY = (
        "This message has been flagged as spam by our automated filter and will "
        "not receive a personalized response. If you believe this is a mistake, "
        "please contact support directly through our verified channels."
    )

    def respond(self, message: str) -> str:
        return self.SPAM_REPLY


class GPTRouter:
    """Simulates routing a message to a GPT-style LLM for a detailed response.

    NOTE: This is a MOCK. No real API call is made. See _call_llm_api for where
    a real OpenAI/Anthropic API call would be inserted if this project were
    connected to a live LLM provider.

    Unlike a purely random mock, this router inspects the message text for
    topic-specific keywords (price, refund, login, crash, etc.) so the
    generated response is at least topically consistent with what the
    customer actually asked - a real LLM would obviously do this far more
    robustly via genuine language understanding, but a keyword-based mock is
    a meaningful improvement over picking a same-class response at random,
    which could pair a question about vehicle pricing with a reply about
    subscription plans just because both fall under "Sales Inquiry".
    """

    # Each entry: (keywords to look for, response to return if any keyword matches).
    # Checked in order - the first matching topic wins.
    SALES_TOPIC_RESPONSES = [
        (["price", "cost", "pricing", "fee", "quote", "how much"],
         "This appears to be a pricing inquiry. Our sales team will provide "
         "detailed price and availability information shortly."),
        (["mileage", "model", "vehicle", "car"],
         "This appears to be a vehicle-related inquiry. We'll retrieve the "
         "requested model, price, and mileage details from our inventory and "
         "follow up shortly."),
        (["subscription", "plan", "upgrade", "downgrade", "tier"],
         "This appears to be a subscription or plan-related inquiry. A "
         "specialist will review your account and respond with personalized "
         "plan options."),
        (["refund", "money back", "reimburse"],
         "Your refund request has been received and is being reviewed by "
         "our finance team. You should hear back within 1-2 business days."),
        (["cancel", "unsubscribe", "cancellation"],
         "We've recorded your cancellation request. A confirmation email "
         "with the final details will be sent to you shortly."),
        (["feature", "suggest", "add", "could you add", "roadmap"],
         "Thank you for your feature suggestion - we've logged it for our "
         "product team to evaluate in an upcoming roadmap review."),
    ]

    COMPLAINT_TOPIC_RESPONSES = [
        (["crash", "bug", "glitch", "broken", "doesn't work", "freeze"],
         "We're sorry to hear you're experiencing this issue. Our technical "
         "team has been notified and will investigate the bug you reported."),
        (["slow", "performance", "lag", "sluggish", "loading"],
         "Thank you for flagging this performance problem - we're looking "
         "into it and will provide an update as soon as we identify the cause."),
        (["login", "log in", "password", "sign in", "credentials", "locked out"],
         "We understand how frustrating login issues can be. Please try "
         "resetting your password, and our team will also review your account."),
        (["sync", "synchron", "syncing"],
         "We apologize for the inconvenience caused by this sync issue. Our "
         "engineers are actively working on a fix and will update you soon."),
        (["security", "hacked", "suspicious", "phishing", "breach"],
         "Thank you for reporting this security concern - we take this "
         "seriously and our security team is reviewing your account activity."),
        (["suspend", "suspension", "blocked", "banned"],
         "We're sorry your account was suspended unexpectedly. We are "
         "escalating this for manual review and will respond with a "
         "resolution shortly."),
    ]

    # Used only if no keyword in the message matched any topic above.
    SALES_FALLBACK = (
        "Thank you for reaching out about our pricing and services. A "
        "member of our sales team will follow up with detailed options shortly."
    )
    COMPLAINT_FALLBACK = (
        "We're sorry to hear about this issue. Our support team has been "
        "notified and will look into it as soon as possible."
    )

    def __init__(self, simulate_latency: bool = True, seed: int = None):
        self.simulate_latency = simulate_latency
        self._rng = random.Random(seed)

    def generate_response(self, message: str, predicted_label: str) -> str:
        """Routes `message` to the (simulated) LLM and returns a generated reply.

        predicted_label: the classifier's predicted class, expected to be
        "Sales Inquiry" or "Complaint" (Spam should never reach this method -
        it's handled by TemplateResponder instead).
        """
        return self._call_llm_api(message, predicted_label)

    def _call_llm_api(self, message: str, predicted_label: str) -> str:
        """Mock LLM call. Replace this method's body with a real API call
        (e.g. OpenAI, Anthropic) to make this a genuine GPT-routing system.
        Example of what a real call might look like:

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a support agent..."},
                    {"role": "user", "content": message},
                ],
            )
            return response.choices[0].message.content

        Current mock behavior: scans `message` for topic keywords and returns
        a topically matching canned response (see SALES_TOPIC_RESPONSES /
        COMPLAINT_TOPIC_RESPONSES), falling back to a generic class-level
        response only if no keyword matches.
        """
        if self.simulate_latency:
            time.sleep(self._rng.uniform(0.05, 0.2))  # simulate network/API latency

        text = message.lower()
        topic_pool = (
            self.SALES_TOPIC_RESPONSES if predicted_label == "Sales Inquiry"
            else self.COMPLAINT_TOPIC_RESPONSES
        )
        fallback = (
            self.SALES_FALLBACK if predicted_label == "Sales Inquiry"
            else self.COMPLAINT_FALLBACK
        )

        for keywords, response in topic_pool:
            if any(keyword in text for keyword in keywords):
                return response

        return fallback


if __name__ == "__main__":
    responder = TemplateResponder()
    router = GPTRouter(seed=42)

    print("--- Spam example ---")
    print(responder.respond("Congratulations! You've won a free gift!"))

    print("\n--- Sales Inquiry: pricing/vehicle example ---")
    print(router.generate_response(
        "Can you tell me the price and mileage of the 2019 model?", "Sales Inquiry"
    ))

    print("\n--- Sales Inquiry: subscription example ---")
    print(router.generate_response(
        "I'd like to upgrade my current plan to the premium tier.", "Sales Inquiry"
    ))

    print("\n--- Complaint: login example ---")
    print(router.generate_response(
        "The app keeps crashing on login.", "Complaint"
    ))

    print("\n--- Complaint: performance example ---")
    print(router.generate_response(
        "Everything has been really slow and sluggish today.", "Complaint"
    ))