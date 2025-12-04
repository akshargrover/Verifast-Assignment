"""
Intent Classification System for Customer Support Messages
Uses LLM API for intelligent intent detection based on conversational context
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import anthropic  # or use openai, langchain, etc.


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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the classifier
        
        Args:
            api_key: API key for Anthropic (or set ANTHROPIC_API_KEY env var)
            model: Model to use for classification
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.system_prompt = self._build_system_prompt()
    
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
        try:
            # Construct the user prompt
            user_prompt = f"""**CONVERSATION HISTORY:**
{conversation_history if conversation_history else "[No prior conversation]"}

**CURRENT CUSTOMER MESSAGE:**
{current_message}

Classify this message and return JSON only."""
            
            # Call the LLM API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract response text
            response_text = response.content[0].text.strip()
            
            # Parse JSON response
            result_dict = self._parse_json_response(response_text)
            
            # Validate and return
            return IntentResult(
                primary=result_dict["primary"],
                secondary=result_dict["secondary"],
                reasoning=result_dict.get("reasoning", "")
            )
            
        except anthropic.APIError as e:
            raise Exception(f"API Error: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {response_text}")
        except KeyError as e:
            raise ValueError(f"Missing required field in response: {str(e)}")
    
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
    
    def classify_batch(
        self, 
        messages: List[Tuple[str, str]]
    ) -> List[IntentResult]:
        """
        Classify multiple messages in batch
        
        Args:
            messages: List of (history, current_message) tuples
        
        Returns:
            List of IntentResult objects
        """
        results = []
        for history, message in messages:
            try:
                result = self.classify(message, history)
                results.append(result)
            except Exception as e:
                # Log error but continue processing
                print(f"Error classifying message '{message[:50]}...': {str(e)}")
                results.append(IntentResult(
                    primary="Error",
                    secondary="classification_failed",
                    reasoning=str(e)
                ))
        return results


# Example usage
def main():
    """
    Example usage of the IntentClassifier
    """
    # Initialize classifier
    classifier = IntentClassifier()
    
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