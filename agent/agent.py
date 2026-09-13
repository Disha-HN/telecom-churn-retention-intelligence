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

Examples:

    "What is the retention summary?"
    "Which customers have the highest predicted churn risk?"
    "Which customers should we prioritize for retention?"

These questions can be answered directly from the structured
retention-intelligence dataset.

The LLM is reserved for questions that require reasoning,
explanation, comparison, or combining multiple pieces of
information.

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

# These tools are available to the LangChain agent.
#
# The fast router can also invoke the deterministic tools
# directly without involving the LLM.

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
The LLM is initialized once when this module is imported.

The LLM is used ONLY for complex questions.

Simple questions are handled by fast_route().
"""

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=TEMPERATURE,
    api_key=GROQ_API_KEY,

    # Limit unnecessary retry delays.
    max_retries=1,

    # Prevent long-running requests from blocking the UI.
    timeout=20,
)


# ============================================================
# LANGCHAIN AGENT
# ============================================================

"""
Create the LangChain agent.

The LangChain agent is the fallback reasoning layer.

Therefore:

    Simple deterministic question
            |
            v
       Direct Tool

    Complex reasoning question
            |
            v
       LangChain + Groq
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
    Convert different tool-result formats into a predictable
    Python representation.

    Supported common outputs include:

        - dict
        - string
        - JSON string
        - dictionary-like objects
    """

    # Already normalized.
    if isinstance(result, dict):
        return result

    # String result.
    if isinstance(result, str):

        text = result.strip()

        # Attempt JSON decoding first.
        try:
            return json.loads(text)

        except Exception:
            return text

    # Generic dictionary-like object.
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

    Returns:
        Customer ID as a string, or None.
    """

    # Match explicit customer references.
    match = re.search(
        r"(?:customer\s*(?:id)?\s*[:#-]?\s*)(\d+)",
        question,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    # Match a standalone numeric identifier.
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
    Extract a requested result count.

    Examples:

        "top 5 high risk customers"
            -> 5

        "show 20 priority customers"
            -> 20

        "highest risk customers"
            -> 10

    The maximum is deliberately limited to 50.
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

    return max(
        1,
        min(value, 50),
    )


# ============================================================
# HELPER: FORMAT PROBABILITY
# ============================================================

def format_probability(
    value,
) -> str:
    """
    Convert a probability value into a readable percentage.

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

    # Convert decimal probability to percentage.
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
    Convert customer-profile tool output into a readable
    Streamlit response.

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

    This is a deterministic presentation layer; the LLM is
    not required for the basic behavioral output.
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
    Format a list returned by the high-risk or priority tools.

    The output is intentionally deterministic so the LLM is
    not required merely to format a ranking result.
    """

    result = normalize_tool_result(result)

    if not isinstance(result, dict):
        return str(result)

    customers = result.get(
        "customers",
        [],
    )

    if not customers:

        return (
            f"### {title}\n\n"
            "No customers were found."
        )

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
            f"- Retention Priority: {priority}"
        )

        lines.append(
            f"- Value: {value}"
        )

        lines.append(
            f"- Behavior: {behavior}"
        )

        lines.append(
            f"- Candidate Action: {action}"
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

    All values come directly from the retention tool.
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

    # Risk distribution.
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

    # Priority distribution.
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

    # Behavioral distribution.
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
    Handle deterministic questions without using Groq.

    This is the primary performance optimization.

    Routing principle:

        Simple question
            |
            v
        Direct tool
            |
            v
        Immediate response

        Complex question
            |
            v
        Return None
            |
            v
        LangChain + Groq

    Returns:
        str:
            When the question can be answered directly.

        None:
            When the question requires the full LangChain agent.
    """

    # --------------------------------------------------------
    # Normalize question
    # --------------------------------------------------------

    q = question.lower().strip()

    # Remove punctuation that could interfere with matching.
    q_clean = re.sub(
        r"[?!.,]+",
        " ",
        q,
    )

    # Normalize repeated whitespace.
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

        return format_summary(
            result
        )

    # --------------------------------------------------------
    # ROUTE 2: HIGHEST PREDICTED CHURN RISK
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # These are deterministic ranking questions.
    #
    # The answer already exists in the
    # predicted_churn_risk field.
    #
    # Therefore:
    #
    #     NO Groq
    #     NO LangChain reasoning
    #
    # This route specifically covers natural-language
    # variations such as:
    #
    #     "Which customers have the highest predicted
    #      churn risk?"
    #
    #     "Who has the highest churn probability?"
    #
    #     "Show the customers most likely to churn."
    #

    high_risk_keywords = [
        "high risk customers",
        "highest risk customers",
        "high-risk customers",
        "very high risk customers",
        "most risky customers",
        "customers at highest risk",
        "customers most likely to churn",
        "most likely customers to churn",
        "likely to churn",

        # Explicit ML-risk wording.
        "highest predicted churn risk",
        "highest predicted churn probability",
        "highest churn risk",
        "highest churn probability",

        "customers with highest churn risk",
        "customers with highest predicted churn risk",
        "customers with highest predicted churn probability",

        "which customers have the highest churn risk",
        "which customers have the highest predicted churn risk",
        "which customers have the highest predicted churn probability",

        "who has the highest churn risk",
        "who has the highest predicted churn risk",

        "who has the highest churn probability",
        "who has the highest predicted churn probability",
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
            f"Top {limit} Highest Churn-Risk Customers",
        )

    # --------------------------------------------------------
    # ROUTE 3: RETENTION PRIORITY CUSTOMERS
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Retention priority is already calculated by the
    # Retention Intelligence layer.
    #
    # The agent should retrieve that decision rather than
    # recompute it or ask the LLM to infer it.
    #
    # This route specifically handles:
    #
    #     "Which customers should we prioritize for retention?"
    #
    # and similar questions.
    #

    priority_keywords = [
        "priority customers",
        "retention priorities",
        "priority 1 customers",
        "priority one customers",
        "customers to retain",
        "customers needing retention",
        "retention targets",

        # Natural-language priority wording.
        "prioritize for retention",
        "should we prioritize",
        "should be prioritized",
        "who should we prioritize",
        "which customers should we prioritize",
        "customers should we prioritize",
        "customers to prioritize",
        "customers we should prioritize",

        "who should be prioritized for retention",
        "which customers should be prioritized for retention",

        "customers needing immediate retention",
        "customers needing immediate action",

        "who needs retention action",
        "who needs immediate retention action",
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

            return format_behavior(
                result
            )

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

            return format_customer_profile(
                result
            )

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
        # the profile is still faster than calling the LLM.
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

    # Returning None sends the question to the full
    # LangChain + Groq reasoning layer.
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

    Different LangChain versions can return different
    response structures, so common formats are handled.
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

        messages = result.get(
            "messages"
        )

        if messages:

            # Process messages backwards because the final
            # AI response is normally the last message.
            for message in reversed(messages):

                # Dictionary-style message.
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

                # Object-style message.
                content = getattr(
                    message,
                    "content",
                    None,
                )

                if content:

                    # Some LangChain versions return content
                    # as a list of blocks.
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

        # Some agent versions return:
        #
        #     {"output": "..."}
        #

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
    Execute the complete retention-agent decision process.

    Processing order:

        1. Validate question
        2. Try deterministic fast routing
        3. If matched, return immediately
        4. Otherwise use LangChain + Groq

    This ensures that simple ranking and lookup queries do
    not incur LLM latency.
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

    except Exception:

        # If deterministic routing fails unexpectedly,
        # fall back to the full agent rather than breaking
        # the application.
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
    #
    # Only questions that cannot be answered confidently
    # through deterministic tools reach this section.
    #
    # Examples:
    #
    #     "Why is customer 2 Priority 2?"
    #
    #     "Compare customers 10 and 20."
    #
    #     "Explain why this customer is risky."
    #
    # These questions genuinely benefit from LLM reasoning.
    #

    try:

        start_time = time.perf_counter()

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

        answer = extract_final_response(
            result
        )

        return answer

    except Exception as exc:

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
    Public interface used by Streamlit.

    Streamlit calls:

        answer = ask_agent(question)

    Keeping this interface simple allows the UI to remain
    independent of the internal routing implementation.
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

    Then enter questions manually.
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