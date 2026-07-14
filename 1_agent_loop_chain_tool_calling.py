from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage,ToolMessage, SystemMessage
from langsmith import traceable


MAX_RETRIES = 10

MODEL = "llama3.2"

@tool
def get_product_price(product: str) -> float:
    """ Lookup the price of the product on the Catalog"""
    print (f">> Executing get_product_price for product {product}")
    prices = {"laptop": 1299, "headphone": 149.08, "keyboard": 89.50 }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """ Apply the discount to the price of the product on the Catalog"""
    discount = {"gold": 5, "silver": 5, "bronze": 5}
    return round((1- discount.get(discount_tier, 0)/100) * price,2)

@traceable(name="Langchain_agent_loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_vertexai"
)
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_RETRIES + 1):
        ai_message = llm_with_tools.invoke(messages)
        toolcalls = ai_message.tool_calls
        if not toolcalls:
            """ No tool calls """
            return ai_message.content
        tool_call = toolcalls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_id = tool_call.get("id")

        tool_to_use = tool_dict.get(tool_name)
        if not tool_to_use:
            raise ValueError(f"Tool {tool_name} not found in tool_dict")
        observation = tool_to_use.invoke(tool_args)
        print(f"Total Result:{observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))

    print("Max Retries Completed")
    return None
if __name__ == "__main__":
    print("Agent loop tool calling")
    result = run_agent("What is the price of the laptop after applying a gold discount")
    print(result)
