"""
Intent Classification System for Customer Support Messages
Uses LLM API for intelligent intent detection based on conversational context
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import dotenv
import google.generativeai as genai

dotenv.load_dotenv()

@dataclass
class IntentResult:
    """Structured output for intent classification"""
    primary: str
    secondary: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class IntentClassifier:
    """
    Intent classification system using LLM with optimized taxonomy
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 2,
        fallback_enabled: bool = True,
    ):
        """
        Initialize the classifier
        
        Args:
            api_key: API key for Google Gemini (or set GOOGLE_API_KEY env var)
            model: Model to use for classification (default: gemini-2.0-flash-exp)
                   Available models: gemini-2.0-flash-exp, gemini-1.5-flash, gemini-1.5-pro
            max_retries: Number of retry attempts for transient failures
            fallback_enabled: Return a safe fallback IntentResult instead of raising
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as GOOGLE_API_KEY environment variable")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model_name=model)
        self.max_retries = max_retries
        self.fallback_enabled = fallback_enabled
        self.system_prompt = self._build_system_prompt()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        self.logger = logging.getLogger(__name__)
    
    def _build_system_prompt(self) -> str:
        """
        Constructs the optimized system prompt with the new intent taxonomy
        """
        return """You are an expert intent classification system for a skincare and haircare e-commerce company called Innovist (also trading as Bare Anatomy and Chemist at Play).

**YOUR TASK:** Classify customer messages into PRIMARY and SECONDARY intent categories based on the COMPLETE conversation history, not just the latest message.

**CRITICAL RULES:**
1. Always consider the FULL conversation context to determine the root intent
2. If a customer shares order details in response to a request, maintain the ORIGINAL intent (e.g., if they're reporting a missing product, it stays "wrong_order" even when they provide order ID)
3. "Basic Interactions" (greetings/acknowledgment) should ONLY be used when the message contains NOTHING else (e.g., just "Hi" or "Ok thanks")
4. If a greeting is combined with a question ("Hi, where is my order?"), classify based on the question, not the greeting
5. "Marked Delivered, Not Received" is ONLY for cases where tracking explicitly shows "Delivered" but customer didn't receive it
6. Each secondary intent MUST belong to its designated primary category - never mix them

**INTENT TAXONOMY:**

PRIMARY: Basic Interactions
├─ greetings: Pure greetings only ("Hi", "Hello", "Good morning")
├─ acknowledgment: Pure acknowledgments only ("Ok", "Thanks", "Got it")
└─ language_preference: Requesting communication in regional language ("Hindi mein bolo", "Tamil please")

PRIMARY: About Company
├─ contact_details: Asking for company phone, email, customer care number
├─ about_team: Questions about the company or team
└─ company_name: "What is your company name?", "Who are you?"

PRIMARY: About Product
├─ product_info: How to use, application instructions, frequency ("How to apply?", "Daily or weekly?")
├─ product_specialty: Benefits, features, what makes it special ("What's special about it?", "Why this product?")
├─ product_usage_sequence: Combining products, order of application ("Use serum before oil?", "Can I use with shampoo?")
├─ results_timeline: When to expect results ("How many days to see results?", "When will hair grow?")
├─ safety_concerns: Age restrictions, pregnancy/breastfeeding safety, side effects, harmful chemicals ("Safe for 10 year old?", "Any side effects?", "Is it chemical-free?")
├─ product_ingredients: Questions about ingredients ("What's in this?", "Does it contain salicylic acid?")
├─ effectiveness: Doubts about whether it works ("Does it really work?", "Is this fake?")
├─ pricing: Price questions ("How much?", "What's the cost?")
├─ comparison: Comparing products ("Which is better X or Y?", "Difference between these two?")
├─ certification: Dermatologist-approved, clinically tested queries
├─ payment_options_available: Payment methods (COD, UPI, etc.)
├─ out_of_stock: Product not available or can't find on website
├─ all_products: General catalog inquiry ("Show all products", "What do you sell?")
├─ place_order: How to place an order, order process
└─ no_query_uc: No further questions ("That's all", "Nothing more")

PRIMARY: Recommendation
├─ seeking_solution: Asking for product recommendations ("What for hair fall?", "Best for oily skin?")
├─ customer_problem: Describing personal issues (acne, dandruff, hair thinning)
└─ customer_information: Sharing details for recommendation (skin type, age, concerns)

PRIMARY: Logistics
├─ order_status: Tracking order, "Where is my order?", delivery status inquiries
├─ order_confirmation: "Is my order confirmed?", "Did my order go through?"
├─ delivery_delay: Complaints about delayed delivery beyond promised timeline
├─ address_change: Request to change delivery address
├─ cancellation: Cancel order requests
├─ wrong_order: Wrong item, damaged item, missing product, exchange requests
├─ order_delivered_but_not_received: Tracking shows "Delivered" but customer didn't receive package
├─ delivery_contact: Request for delivery person's phone number
├─ refund_required: Asking for refund to be processed
├─ refund_policy: Refund status, timeline, how refunds work
├─ return_policy: Return conditions, how to return products
├─ payment: Payment issues (failed payment, money deducted, payment errors)
├─ discount: Offers, promo codes, discounts ("Any offers?", "Discount code?")
├─ ecommerce_quick_commerce: Availability on Amazon, Flipkart, etc.
├─ b2b_queries: Bulk orders, distribution, wholesale
├─ website_issue: Website errors, can't place order, site not working
├─ account_management: Delete account, update phone/email, profile changes
├─ talk_to_agent: Explicit request to speak with human ("Connect me to agent", "I want to talk to someone")
├─ service_complaint: Complaints about delivery service, customer support, company service
├─ product_complaint: Product didn't work, caused irritation, poor results, dissatisfaction with product performance
└─ feedback: General positive/negative feedback (not rising to complaint level)

PRIMARY: Special Categories
├─ gibberish: Unintelligible messages, random characters ("Xxxxxxxxxx", "ðŸ˜¡ðŸ˜¡ðŸ˜¡")
└─ out_of_scope: Requests completely unrelated to skincare/haircare business

**OUTPUT FORMAT:**
Return ONLY valid JSON in this exact format:
{
  "primary": "<primary_category>",
  "secondary": "<secondary_category>",
  "reasoning": "<brief explanation of why this classification>"
}

**EXAMPLES:**

Message: "Hi, where is my order?"
Correct: {"primary": "Logistics", "secondary": "order_status", "reasoning": "Customer asking about order location despite greeting"}
Wrong: {"primary": "Basic Interactions", "secondary": "greetings"}

Message: "Hindi mein batao"
Correct: {"primary": "Basic Interactions", "secondary": "language_preference", "reasoning": "Customer requesting Hindi language communication"}

Message: "Can I use serum before oil?"
Correct: {"primary": "About Product", "secondary": "product_usage_sequence", "reasoning": "Asking about order of product application"}

Message: "Is this safe for my 10 year old daughter?"
Correct: {"primary": "About Product", "secondary": "safety_concerns", "reasoning": "Concerned about age appropriateness"}

Message: "How many days to see results?"
Correct: {"primary": "About Product", "secondary": "results_timeline", "reasoning": "Asking about expected timeline for visible results"}

Message: "9876543210" (after agent asked for contact to track order)
Correct: {"primary": "Logistics", "secondary": "order_status", "reasoning": "Providing contact info to resolve order tracking issue - root intent is order status"}

Message: "Tracking shows delivered but I didn't get it"
Correct: {"primary": "Logistics", "secondary": "order_delivered_but_not_received", "reasoning": "Explicitly states tracking shows delivered but not received"}

Message: "My order is delayed"
Correct: {"primary": "Logistics", "secondary": "delivery_delay", "reasoning": "Complaint about delivery taking too long"}

Message: "This product didn't work for me at all"
Correct: {"primary": "Logistics", "secondary": "product_complaint", "reasoning": "Dissatisfaction with product performance"}

Message: "Your delivery person was rude"
Correct: {"primary": "Logistics", "secondary": "service_complaint", "reasoning": "Complaint about service experience"}

Message: "Xxxxxxxxxx"
Correct: {"primary": "Special Categories", "secondary": "gibberish", "reasoning": "Unintelligible message"}

Now classify the following message based on the conversation history provided.
"""
    
    def classify(
        self, 
        current_message: str, 
        conversation_history: Optional[str] = ""
    ) -> IntentResult:
        """
        Classify a customer message into intent categories
        
        Args:
            current_message: The latest message from the customer
            conversation_history: Previous conversation context (optional but recommended)
        
        Returns:
            IntentResult object with primary, secondary, and reasoning
        
        Raises:
            ValueError: If API returns invalid response
            Exception: For API errors
        """
        response_text = ""
        attempts = 0
        last_error: Optional[Exception] = None

        while attempts <= self.max_retries:
            try:
                # Construct the full prompt (system + user)
                user_prompt = f"""**CONVERSATION HISTORY:**
{conversation_history if conversation_history else "[No prior conversation]"}

**CURRENT CUSTOMER MESSAGE:**
{current_message}

Classify this message and return JSON only."""
                
                # Combine system prompt and user prompt for Gemini
                full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
                
                # Call the Gemini API
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1000,
                        temperature=0.1
                    )
                )
                
                # Extract response text
                response_text = response.text.strip()
                
                # Parse JSON response
                result_dict = self._parse_json_response(response_text)
                
                # Validate and return
                return IntentResult(
                    primary=result_dict["primary"],
                    secondary=result_dict["secondary"],
                    reasoning=result_dict.get("reasoning", "")
                )
                
            except (json.JSONDecodeError, KeyError) as e:
                last_error = e
                attempts += 1
                self.logger.warning("Parsing failed (attempt %s/%s): %s", attempts, self.max_retries, str(e))
            except Exception as e:
                last_error = e
                attempts += 1
                self.logger.warning("Classification failed (attempt %s/%s): %s", attempts, self.max_retries, str(e))

        # If we are here, all attempts failed
        if self.fallback_enabled:
            self.logger.error("Returning fallback result after failure: %s", str(last_error))
            return self._build_fallback_result(
                str(last_error) if last_error else "Unknown error",
                current_message=current_message,
                conversation_history=conversation_history,
            )

        # Fallback disabled: raise the last error
        if last_error:
            if isinstance(last_error, json.JSONDecodeError):
                raise ValueError(f"Failed to parse JSON response: {response_text}")
            if isinstance(last_error, KeyError):
                raise ValueError(f"Missing required field in response: {str(last_error)}")
            if "API" in str(last_error) or "api" in str(last_error).lower():
                raise Exception(f"API Error: {str(last_error)}")
            raise Exception(f"Error: {str(last_error)}")
        raise Exception("Unknown classification error")
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Parse JSON from LLM response, handling markdown code blocks
        
        Args:
            response_text: Raw text from LLM
        
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        return json.loads(response_text)

    def _build_fallback_result(
        self,
        error_reason: str,
        current_message: Optional[str] = None,
        conversation_history: Optional[str] = None
    ) -> IntentResult:
        """
        Build a safe fallback intent result when classification fails.
        Tries a lightweight rule-based match before defaulting to out_of_scope.
        """
        if current_message:
            rule_match = self._rule_based_intent(current_message, conversation_history or "")
            if rule_match:
                primary, secondary = rule_match
                return IntentResult(
                    primary=primary,
                    secondary=secondary,
                    reasoning=f"Rule-based fallback classification after error: {error_reason}"
                )

        return IntentResult(
            primary="Special Categories",
            secondary="out_of_scope",
            reasoning=f"Fallback classification used due to error: {error_reason}"
        )

    def _rule_based_intent(
        self,
        current_message: str,
        conversation_history: str
    ) -> Optional[Tuple[str, str]]:
        """
        Simple keyword-based fallback classifier.
        Returns (primary, secondary) or None if no rule matches.
        """
        text = f"{conversation_history} {current_message}".lower()

        rules: Dict[Tuple[str, str], List[str]] = {
            ("Logistics", "order_status"): [
                "where is my order",
                "track",
                "tracking",
                "order status",
                "not received",
                "not delivered",
                "pending",
                "kaha hai",
            ],
            ("Logistics", "order_delivered_but_not_received"): [
                "marked delivered",
                "shows delivered",
                "status delivered",
                "delivered but not received",
            ],
            ("Logistics", "delivery_delay"): [
                "delayed",
                "delay",
                "late",
            ],
            ("Logistics", "wrong_order"): [
                "wrong item",
                "damaged",
                "broken",
                "missing",
                "exchange",
                "return",
            ],
            ("Basic Interactions", "greetings"): [
                "hi",
                "hello",
                "hey",
            ],
            ("Basic Interactions", "acknowledgment"): [
                "ok",
                "okay",
                "thanks",
                "thank you",
            ],
            ("Basic Interactions", "language_preference"): [
                "hindi",
                "tamil",
                "telugu",
                "kannada",
                "language",
            ],
        }

        for (primary, secondary), keywords in rules.items():
            if any(keyword in text for keyword in keywords):
                return primary, secondary

        return None
    
    def classify_batch(
        self, 
        messages: List[Tuple[str, str]],
        parallel: bool = True,
        max_workers: int = 4
    ) -> List[IntentResult]:
        """
        Classify multiple messages in batch
        
        Args:
            messages: List of (history, current_message) tuples
            parallel: Enable parallel processing for higher throughput
            max_workers: Max worker threads when parallel is True
        
        Returns:
            List of IntentResult objects
        """
        if not parallel or len(messages) == 1:
            return [self.classify(message, history) for history, message in messages]

        results: List[Optional[IntentResult]] = [None] * len(messages)

        def _worker(idx: int, history: str, message: str) -> None:
            try:
                results[idx] = self.classify(message, history)
            except Exception as e:
                self.logger.error("Error classifying message '%s': %s", message[:50], str(e))
                results[idx] = self._build_fallback_result(
                    str(e),
                    current_message=message,
                    conversation_history=history,
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, idx, history, message): idx
                for idx, (history, message) in enumerate(messages)
            }
            for future in as_completed(futures):
                # Exceptions are handled inside _worker; this ensures all tasks complete
                future.result()

        # All slots should be filled
        return [res for res in results if res is not None]


# Example usage
def main():
    """
    Example usage of the IntentClassifier
    """
    # Initialize classifier
    classifier = IntentClassifier(model="gemini-2.5-flash")
    
    # Example 1: Simple order tracking
    result1 = classifier.classify(
        current_message="Where is my order?",
        conversation_history=""
    )
    print("Example 1:")
    print(f"Primary: {result1.primary}")
    print(f"Secondary: {result1.secondary}")
    print(f"Reasoning: {result1.reasoning}\n")
    
    # Example 2: With conversation history
    history = """human: How to track order?
ai: I'm sorry, but I can't find any information associated with your order. Could you please share your contact number or email ID so I can assist you further?"""
    
    result2 = classifier.classify(
        current_message="rohanvijay990@gmail.com 9380163998",
        conversation_history=history
    )
    print("Example 2:")
    print(f"Primary: {result2.primary}")
    print(f"Secondary: {result2.secondary}")
    print(f"Reasoning: {result2.reasoning}\n")
    
    # Example 3: Language preference
    result3 = classifier.classify(
        current_message="Hindi mein bolo",
        conversation_history=""
    )
    print("Example 3:")
    print(f"Primary: {result3.primary}")
    print(f"Secondary: {result3.secondary}")
    print(f"Reasoning: {result3.reasoning}\n")
    
    # Example 4: Product usage sequence
    result4 = classifier.classify(
        current_message="Can I use serum with oil?",
        conversation_history="human: How to use it in daily routine? Rosemary & Hibiscus Hair Growth Oil"
    )
    print("Example 4:")
    print(f"Primary: {result4.primary}")
    print(f"Secondary: {result4.secondary}")
    print(f"Reasoning: {result4.reasoning}\n")


if __name__ == "__main__":
    main()