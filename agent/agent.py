"""
Telecom Customer Churn Retention AI Agent
==========================================

This module connects:

    Streamlit UI
          |
          v
    Fast Query Router
          |
     +----+----------------------+
     |                           |
 Simple questions          Complex questions
     |                           |
     v                           v
Retention Tools             LangChain Agent
     |                           |
     |                           v
     |                       Groq LLM
     |                           |
     +------------+--------------+
                  |
                  v
             Final Answer


IMPORTANT DESIGN PRINCIPLE
--------------------------

Simple questions should NOT call the LLM.

For example:

    "summary of retention"

should directly call:

    get_retention_summary()

instead of:

    User
      -> Groq
      -> Tool
      -> Groq
      -> Answer


This significantly reduces response time.

The LLM is reserved for questions that actually require
reasoning or combining multiple pieces of information.


SOURCE OF TRUTH
---------------

ML churn probability:
    predicted_churn_risk

Retention intelligence:
    risk_tier
    retention_priority
    behavioral_state
    behavioral_evidence
    candidate_action
    decision_rationale
    value_tier

The binary dataset field:

    churn_probability

is NOT treated as the ML probability.
"""


# ============================================================
# IMPORTS
# ============================================================

import json
import re
import time
from typing import Optional


# LangChain / Groq
from langchain_groq import ChatGroq
from langchain.agents import create_agent


# Project configuration
from agent.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TEMPERATURE,
)


# System prompt used by the complex LangChain agent
from agent.prompts import SYSTEM_PROMPT


# Retention intelligence tools
from agent.tool import (
    get_customer_profile,
    get_high_risk_customers,
    get_priority_customers,
    analyze_customer_behavior,
    get_retention_summary,
)


# ============================================================
# TOOL COLLECTION
# ============================================================

# These are the tools available to the LangChain agent.
#
# IMPORTANT:
# The fast router below can call these tools directly without
# involving the LLM.

TOOLS = [
    get_customer_profile,
    get_high_risk_customers,
    get_priority_customers,
    analyze_customer_behavior,
    get_retention_summary,
]


# ============================================================
# GROQ LLM CONFIGURATION
# ============================================================

"""
The LLM is created only once when this module is imported.

The model is used ONLY for complex questions.

Simple questions are handled by fast_route().
"""

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=TEMPERATURE,
    api_key=GROQ_API_KEY,

    # Prevent unnecessary retries from making the UI feel slow.
    max_retries=1,

    # Prevent the request from hanging for a very long time.
    timeout=20,
)


# ============================================================
# LANGCHAIN AGENT
# ============================================================

"""
Create the LangChain agent.

This agent is used only when the fast router cannot confidently
handle the question.

Therefore:

    Simple question
        -> direct tool

    Complex question
        -> LangChain + Groq
"""

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# HELPER: NORMALIZE TOOL RESULT
# ============================================================

def normalize_tool_result(result):
    """
    Convert a LangChain tool result into a normal Python object.

    Different LangChain versions may return:
        - dict
        - string
        - JSON string
        - other serializable objects

    This function makes the rest of the code easier to handle.
    """

    # Already a dictionary.
    if isinstance(result, dict):
        return result

    # Already a string.
    if isinstance(result, str):

        text = result.strip()

        # Try JSON decoding.
        try:
            return json.loads(text)

        except Exception:
            return text

    # Try converting other objects into a dictionary.
    try:
        return dict(result)

    except Exception:
        return str(result)


# ============================================================
# HELPER: EXTRACT CUSTOMER ID
# ============================================================

def extract_customer_id(
    question: str,
) -> Optional[str]:
    """
    Extract a customer ID from a natural-language question.

    Examples:

        "What is the risk for customer 12345?"
            -> "12345"

        "Tell me about customer 9876"
            -> "9876"

        "Why is 54321 at risk?"
            -> "54321"

    The function intentionally looks for a numeric ID.

    Returns:
        Customer ID as string, or None.
    """

    # Look for phrases such as:
    #
    # customer 12345
    # customer ID 12345
    # customer id: 12345

    match = re.search(
        r"(?:customer\s*(?:id)?\s*[:#-]?\s*)(\d+)",
        question,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    # If no explicit "customer" keyword exists,
    # look for a standalone numeric identifier.
    #
    # This is useful for questions such as:
    #
    # "12345 churn probability"

    standalone = re.search(
        r"\b(\d{3,})\b",
        question,
    )

    if standalone:
        return standalone.group(1)

    return None


# ============================================================
# HELPER: EXTRACT TOP N
# ============================================================

def extract_top_n(
    question: str,
    default: int = 10,
) -> int:
    """
    Extract a requested number from the question.

    Examples:

        "top 5 high risk customers"
            -> 5

        "show 20 priority customers"
            -> 20

        "highest risk customers"
            -> 10

    The maximum is deliberately limited to 50 so that very
    large responses are not sent to the LLM.
    """

    match = re.search(
        r"\b(?:top|first|show|list|give)\s+(\d+)\b",
        question,
        re.IGNORECASE,
    )

    if not match:
        return default

    try:
        value = int(match.group(1))

    except ValueError:
        return default

    # Keep requests inside a safe range.
    return max(1, min(value, 50))


# ============================================================
# HELPER: FORMAT PROBABILITY
# ============================================================

def format_probability(value) -> str:
    """
    Convert a probability into a readable percentage.

    Examples:

        0.986
            -> 98.60%

        98.6
            -> 98.60%
    """

    try:
        numeric = float(value)

    except (TypeError, ValueError):
        return "N/A"

    # Decimal probability.
    if 0 <= numeric <= 1:
        numeric *= 100

    return f"{numeric:.2f}%"


# ============================================================
# HELPER: FORMAT CUSTOMER PROFILE
# ============================================================

def format_customer_profile(
    result: dict,
) -> str:
    """
    Convert the customer profile tool result into a clean
    human-readable response.

    No LLM is required for this formatting.
    """

    result = normalize_tool_result(result)

    if not isinstance(result, dict):
        return str(result)

    if result.get("status") == "not_found":
        return result.get(
            "message",
            "Customer was not found.",
        )

    customer_id = result.get(
        "customer_id",
        "Unknown",
    )

    probability = format_probability(
        result.get("ml_churn_probability")
    )

    risk = result.get(
        "risk_tier",
        "Unknown",
    )

    priority = result.get(
        "retention_priority",
        "Unknown",
    )

    behavior = result.get(
        "behavioral_state",
        "Unknown",
    )

    value = result.get(
        "value_tier",
        "Unknown",
    )

    arpu = result.get(
        "arpu_8",
        "N/A",
    )

    action = result.get(
        "candidate_action",
        "N/A",
    )

    rationale = result.get(
        "decision_rationale",
        "N/A",
    )

    evidence = result.get(
        "behavioral_evidence",
        "N/A",
    )

    recent = result.get(
        "recent_deterioration_count",
        0,
    )

    persistent = result.get(
        "persistent_deterioration_count",
        0,
    )

    coordinated = result.get(
        "coordinated_deterioration_count",
        0,
    )

    return (
        f"### Customer {customer_id}\n\n"
        f"**ML Churn Probability:** {probability}\n\n"
        f"**Risk Tier:** {risk}\n\n"
        f"**Retention Priority:** {priority}\n\n"
        f"**Customer Value:** {value}\n\n"
        f"**Behavioral State:** {behavior}\n\n"
        f"**ARPU (Month 8):** {arpu}\n\n"
        f"**Behavioral Evidence:** {evidence}\n\n"
        f"**Recent Deterioration Count:** {recent}\n\n"
        f"**Persistent Deterioration Count:** {persistent}\n\n"
        f"**Coordinated Deterioration Count:** {coordinated}\n\n"
        f"**Candidate Retention Action:** {action}\n\n"
        f"**Decision Rationale:** {rationale}"
    )


# ============================================================
# HELPER: FORMAT BEHAVIOR
# ============================================================

def format_behavior(
    result: dict,
) -> str:
    """
    Format behavioral-analysis information.
    """

    result = normalize_tool_result(result)

    if not isinstance(result, dict):
        return str(result)

    if result.get("status") == "not_found":
        return result.get(
            "message",
            "Customer was not found.",
        )

    customer_id = result.get(
        "customer_id",
        "Unknown",
    )

    probability = format_probability(
        result.get("ml_churn_probability")
    )

    risk = result.get(
        "risk_tier",
        "Unknown",
    )

    priority = result.get(
        "retention_priority",
        "Unknown",
    )

    behavior = result.get(
        "behavioral_state",
        "Unknown",
    )

    evidence = result.get(
        "behavioral_evidence",
        "N/A",
    )

    recent = result.get(
        "recent_deterioration_count",
        0,
    )

    persistent = result.get(
        "persistent_deterioration_count",
        0,
    )

    coordinated = result.get(
        "coordinated_deterioration_count",
        0,
    )

    action = result.get(
        "candidate_action",
        "N/A",
    )

    rationale = result.get(
        "decision_rationale",
        "N/A",
    )

    return (
        f"### Behavioral Analysis — Customer {customer_id}\n\n"
        f"**ML Churn Probability:** {probability}\n\n"
        f"**Risk Tier:** {risk}\n\n"
        f"**Retention Priority:** {priority}\n\n"
        f"**Behavioral State:** {behavior}\n\n"
        f"**Behavioral Evidence:** {evidence}\n\n"
        f"**Recent Deterioration:** {recent}\n\n"
        f"**Persistent Deterioration:** {persistent}\n\n"
        f"**Coordinated Deterioration:** {coordinated}\n\n"
        f"**Candidate Action:** {action}\n\n"
        f"**Decision Rationale:** {rationale}"
    )


# ============================================================
# HELPER: FORMAT CUSTOMER LIST
# ============================================================

def format_customer_list(
    result: dict,
    title: str,
) -> str:
    """
    Format high-risk or priority customer results.

    The formatting is intentionally simple so the result can
    be displayed directly inside Streamlit.
    """

    result = normalize_tool_result(result)

    if not isinstance(result, dict):
        return str(result)

    customers = result.get(
        "customers",
        [],
    )

    if not customers:
        return f"### {title}\n\nNo customers were found."

    lines = [
        f"### {title}",
        "",
    ]

    for index, customer in enumerate(
        customers,
        start=1,
    ):

        customer_id = customer.get(
            "customer_id",
            "Unknown",
        )

        probability = format_probability(
            customer.get(
                "ml_churn_probability"
            )
        )

        risk = customer.get(
            "risk_tier",
            "Unknown",
        )

        priority = customer.get(
            "retention_priority",
            "Unknown",
        )

        value = customer.get(
            "value_tier",
            "Unknown",
        )

        behavior = customer.get(
            "behavioral_state",
            "Unknown",
        )

        action = customer.get(
            "candidate_action",
            "N/A",
        )

        lines.append(
            f"**{index}. Customer {customer_id}**"
        )

        lines.append(
            f"- ML Churn Probability: {probability}"
        )

        lines.append(
            f"- Risk: {risk}"
        )

        lines.append(
            f"- Priority: {priority}"
        )

        lines.append(
            f"- Value: {value}"
        )

        lines.append(
            f"- Behavior: {behavior}"
        )

        lines.append(
            f"- Action: {action}"
        )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# HELPER: FORMAT RETENTION SUMMARY
# ============================================================

def format_summary(
    result: dict,
) -> str:
    """
    Format the overall retention summary.

    All numbers come directly from the retention tool.

    The LLM is NOT involved.
    """

    result = normalize_tool_result(result)

    if not isinstance(result, dict):
        return str(result)

    if result.get("status") != "success":
        return str(result)

    total = result.get(
        "total_customers",
        0,
    )

    average = format_probability(
        result.get(
            "average_ml_churn_probability"
        )
    )

    high_risk = result.get(
        "high_or_very_high_risk_customers",
        0,
    )

    high_risk_pct = result.get(
        "high_or_very_high_risk_percentage",
        0,
    )

    priority_1 = result.get(
        "priority_1_customers",
        0,
    )

    priority_1_pct = result.get(
        "priority_1_percentage",
        0,
    )

    risk_distribution = result.get(
        "risk_distribution",
        {},
    )

    priority_distribution = result.get(
        "priority_distribution",
        {},
    )

    behavioral_distribution = result.get(
        "behavioral_state_distribution",
        {},
    )

    lines = [
        "### Telecom Retention Summary",
        "",
        f"**Total Customers:** {total:,}",
        "",
        f"**Average ML Churn Probability:** {average}",
        "",
        (
            f"**High / Very High Risk Customers:** "
            f"{high_risk:,} ({high_risk_pct:.2f}%)"
        ),
        "",
        (
            f"**Priority 1 Customers:** "
            f"{priority_1:,} ({priority_1_pct:.2f}%)"
        ),
        "",
        "### Risk Distribution",
        "",
    ]

    # Add risk distribution.
    for risk, count in risk_distribution.items():

        try:
            count_text = f"{int(count):,}"

        except (TypeError, ValueError):
            count_text = str(count)

        lines.append(
            f"- **{risk}:** {count_text}"
        )

    lines.extend(
        [
            "",
            "### Retention Priority Distribution",
            "",
        ]
    )

    # Add priority distribution.
    for priority, count in priority_distribution.items():

        try:
            count_text = f"{int(count):,}"

        except (TypeError, ValueError):
            count_text = str(count)

        lines.append(
            f"- **{priority}:** {count_text}"
        )

    lines.extend(
        [
            "",
            "### Behavioral State Distribution",
            "",
        ]
    )

    # Add behavioral distribution.
    for behavior, count in behavioral_distribution.items():

        try:
            count_text = f"{int(count):,}"

        except (TypeError, ValueError):
            count_text = str(count)

        lines.append(
            f"- **{behavior}:** {count_text}"
        )

    return "\n".join(lines)


# ============================================================
# FAST ROUTER
# ============================================================

def fast_route(
    question: str,
) -> Optional[str]:
    """
    Handle simple deterministic questions without using Groq.

    This function is the MAIN PERFORMANCE OPTIMIZATION.

    It returns:
        str
            When the question can be answered directly.

        None
            When the question requires the full LangChain agent.
    """

    # --------------------------------------------------------
    # Normalize the question
    # --------------------------------------------------------

    q = question.lower().strip()

    # Remove unnecessary punctuation.
    q_clean = re.sub(
        r"[?!.,]+",
        " ",
        q,
    )

    # Normalize repeated spaces.
    q_clean = re.sub(
        r"\s+",
        " ",
        q_clean,
    ).strip()

    # --------------------------------------------------------
    # ROUTE 1: RETENTION SUMMARY
    # --------------------------------------------------------

    summary_keywords = [
        "summary of retention",
        "retention summary",
        "overall retention",
        "retention overview",
        "overall churn",
        "overall risk",
        "dataset summary",
        "customer summary",
        "retention statistics",
        "churn statistics",
        "summary",
    ]

    if any(
        keyword in q_clean
        for keyword in summary_keywords
    ):

        result = get_retention_summary.invoke({})

        return format_summary(result)

    # --------------------------------------------------------
    # ROUTE 2: HIGH-RISK CUSTOMERS
    # --------------------------------------------------------

    high_risk_keywords = [
        "high risk customers",
        "highest risk customers",
        "high-risk customers",
        "very high risk customers",
        "most risky customers",
        "customers at highest risk",
        "customers most likely to churn",
        "likely to churn",
    ]

    if any(
        keyword in q_clean
        for keyword in high_risk_keywords
    ):

        limit = extract_top_n(
            question,
            default=10,
        )

        result = get_high_risk_customers.invoke(
            {
                "limit": limit,
            }
        )

        return format_customer_list(
            result,
            f"Top {limit} High-Risk Customers",
        )

    # --------------------------------------------------------
    # ROUTE 3: PRIORITY CUSTOMERS
    # --------------------------------------------------------

    priority_keywords = [
        "priority customers",
        "retention priorities",
        "priority 1 customers",
        "priority one customers",
        "customers to retain",
        "customers needing retention",
        "retention targets",
    ]

    if any(
        keyword in q_clean
        for keyword in priority_keywords
    ):

        limit = extract_top_n(
            question,
            default=10,
        )

        result = get_priority_customers.invoke(
            {
                "limit": limit,
            }
        )

        return format_customer_list(
            result,
            f"Top {limit} Retention Priority Customers",
        )

    # --------------------------------------------------------
    # ROUTE 4: CUSTOMER-SPECIFIC QUESTIONS
    # --------------------------------------------------------

    customer_id = extract_customer_id(
        question
    )

    if customer_id:

        # ----------------------------------------------------
        # Behavioral / why-at-risk questions
        # ----------------------------------------------------

        behavior_keywords = [
            "why",
            "behavior",
            "behaviour",
            "reason",
            "risk reason",
            "at risk",
            "risk factors",
            "deterioration",
            "decline",
            "signals",
            "evidence",
        ]

        if any(
            keyword in q_clean
            for keyword in behavior_keywords
        ):

            result = analyze_customer_behavior.invoke(
                {
                    "customer_id": customer_id,
                }
            )

            return format_behavior(result)

        # ----------------------------------------------------
        # Profile questions
        # ----------------------------------------------------

        profile_keywords = [
            "profile",
            "details",
            "information",
            "tell me about",
            "show customer",
            "customer information",
            "customer details",
        ]

        if any(
            keyword in q_clean
            for keyword in profile_keywords
        ):

            result = get_customer_profile.invoke(
                {
                    "customer_id": customer_id,
                }
            )

            return format_customer_profile(result)

        # ----------------------------------------------------
        # Direct churn probability questions
        # ----------------------------------------------------

        probability_keywords = [
            "churn probability",
            "churn risk",
            "probability of churn",
            "risk percentage",
            "risk percent",
            "likelihood of churn",
            "chance of churn",
        ]

        if any(
            keyword in q_clean
            for keyword in probability_keywords
        ):

            result = get_customer_profile.invoke(
                {
                    "customer_id": customer_id,
                }
            )

            result = normalize_tool_result(
                result
            )

            if result.get("status") == "not_found":

                return result.get(
                    "message",
                    "Customer was not found.",
                )

            probability = format_probability(
                result.get(
                    "ml_churn_probability"
                )
            )

            risk = result.get(
                "risk_tier",
                "Unknown",
            )

            return (
                f"### Customer {customer_id}\n\n"
                f"**ML Churn Probability:** "
                f"{probability}\n\n"
                f"**Risk Tier:** {risk}"
            )

        # ----------------------------------------------------
        # Direct customer question
        # ----------------------------------------------------
        #
        # If a question clearly refers to one customer but
        # doesn't match a more specific category, retrieving
        # the profile is still faster and safer than calling
        # the LLM.
        #

        direct_customer_keywords = [
            "customer",
            "about",
            "who is",
            "status",
        ]

        if any(
            keyword in q_clean
            for keyword in direct_customer_keywords
        ):

            result = get_customer_profile.invoke(
                {
                    "customer_id": customer_id,
                }
            )

            return format_customer_profile(
                result
            )

    # --------------------------------------------------------
    # NO FAST ROUTE
    # --------------------------------------------------------

    # Returning None tells run_full_agent() to use
    # LangChain + Groq.
    return None


# ============================================================
# HELPER: EXTRACT FINAL RESPONSE
# ============================================================

def extract_final_response(
    result,
) -> str:
    """
    Extract the final assistant response from a LangChain
    agent invocation.

    LangChain versions may return different structures,
    so this function handles common formats.
    """

    # --------------------------------------------------------
    # Direct string
    # --------------------------------------------------------

    if isinstance(result, str):
        return result.strip()

    # --------------------------------------------------------
    # Dictionary response
    # --------------------------------------------------------

    if isinstance(result, dict):

        # Common LangGraph/LangChain structure:
        #
        # {
        #     "messages": [...]
        # }

        messages = result.get(
            "messages"
        )

        if messages:

            # Process messages from the end because the
            # final AI message is normally last.
            for message in reversed(messages):

                # Dictionary message.
                if isinstance(
                    message,
                    dict,
                ):

                    content = message.get(
                        "content"
                    )

                    if content:
                        return str(
                            content
                        ).strip()

                # Object message.
                content = getattr(
                    message,
                    "content",
                    None,
                )

                if content:

                    # Some LangChain responses may return
                    # a list of content blocks.
                    if isinstance(
                        content,
                        list,
                    ):

                        text_parts = []

                        for block in content:

                            if isinstance(
                                block,
                                dict,
                            ):

                                text = block.get(
                                    "text"
                                )

                                if text:
                                    text_parts.append(
                                        str(text)
                                    )

                            else:
                                text_parts.append(
                                    str(block)
                                )

                        if text_parts:
                            return "\n".join(
                                text_parts
                            ).strip()

                    return str(
                        content
                    ).strip()

        # Some agent versions may return:
        #
        # {"output": "..."}
        output = result.get(
            "output"
        )

        if output:
            return str(
                output
            ).strip()

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    return str(result)


# ============================================================
# FULL AGENT EXECUTION
# ============================================================

def run_full_agent(
    question: str,
) -> str:
    """
    Execute the complete decision process.

    First:

        fast_route()

    is attempted.

    If the question is simple:
        -> return immediately

    Otherwise:
        -> LangChain agent
        -> Groq
        -> tools
        -> final answer
    """

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    question = str(
        question
    ).strip()

    if not question:

        return (
            "Please enter a question about "
            "customer churn or retention."
        )

    # --------------------------------------------------------
    # STEP 1: FAST ROUTING
    # --------------------------------------------------------

    start_time = time.perf_counter()

    try:

        fast_answer = fast_route(
            question
        )

    except Exception as exc:

        # If the fast router itself fails, do not silently
        # hide the error.
        #
        # However, we can still attempt the full agent.
        fast_answer = None

    fast_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # STEP 2: RETURN DIRECT ANSWER
    # --------------------------------------------------------

    if fast_answer is not None:

        return fast_answer

    # --------------------------------------------------------
    # STEP 3: COMPLEX QUESTION
    # --------------------------------------------------------

    """
    Only questions that cannot be confidently answered by
    the deterministic tools reach this section.

    This is where Groq/LangChain is intentionally used.
    """

    try:

        start_time = time.perf_counter()

        # LangChain agent invocation.
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # Extract the final answer.
        answer = extract_final_response(
            result
        )

        return answer

    except Exception as exc:

        # Return a clean error to Streamlit instead of exposing
        # a large traceback to the end user.
        return (
            "### Agent Error\n\n"
            f"{exc}"
        )


# ============================================================
# PUBLIC FUNCTION USED BY STREAMLIT
# ============================================================

def ask_agent(
    question: str,
) -> str:
    """
    Public interface used by the Streamlit application.

    Streamlit should call:

        answer = ask_agent(question)

    This function keeps the Streamlit code simple.
    """

    return run_full_agent(
        question
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main():
    """
    Simple command-line test.

    Run:

        python -m agent.agent

    Then type questions manually.
    """

    print()
    print("=" * 60)
    print("Telecom Retention AI Agent")
    print("=" * 60)
    print()
    print("Type 'exit' to stop.")
    print()

    while True:

        try:

            question = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            break

        if question.lower() in {
            "exit",
            "quit",
        }:

            break

        if not question:
            continue

        print()
        print("Agent:")

        answer = ask_agent(
            question
        )

        print(answer)
        print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
