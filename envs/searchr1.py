import re
import requests
import os
import string
from typing import Union, List


SYSTEM_PROMPT = """
# COMPREHENSIVE REPORT EVALUATION RUBRIC

## EVALUATION INSTRUCTIONS
You are an expert evaluator assessing the quality of a generated comprehensive research report, inspired by the STORM process (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking). Evaluate ONLY the final report output, inferring adherence to STORM stages from its content and structure. Do not consider any intermediate steps or process details—focus solely on the report itself.

The STORM process implies the report should demonstrate:
- *Pre-writing Research*: Evidence of deep, multi-perspective research (e.g., 5-7 diverse angles like historical, technical, economic, societal, ethical) with at least 10 cited sources, grounded claims, and synthesis of information (inferred from content variety and citations).
- *Outline Generation*: A logical, hierarchical structure organizing the information (e.g., Introduction, Main sections with subsections, Conclusion, References).
- *Article Generation*: A full-length, neutral, encyclopedic article (2000-5000 words) with inline citations [1] linking to sources.
- *Polishing*: Refined presentation with an executive summary, no redundancies, coherent flow, and verified claims (no unsubstantiated info).

## EVALUATION CRITERIA
Score each criterion quantitatively on a scale of 0-10 (0 = Completely absent/poor, 10 = Exemplary). Provide a brief justification (1-2 sentences) for each score. Then, compute a total score (average of all criteria scores, rounded to one decimal place) and an overall classification.

### 1. RESEARCH DEPTH & BREADTH (Pre-writing Stage)
- Score based on: Evidence of multi-perspective coverage (e.g., diverse angles explored thoroughly), at least 10 diverse, relevant sources cited, and synthesis of information without gaps or superficiality. All claims must appear grounded in citations.
- [ ] Covers 5-7+ perspectives (e.g., historical, technical, economic, societal, ethical)
- [ ] At least 10 unique, verifiable sources (diverse types: academic, news, expert opinions)
- [ ] No unsubstantiated claims or hallucinations

### 2. STRUCTURE & ORGANIZATION (Outline Generation Stage)
- Score based on: Clear, hierarchical structure with logical flow (e.g., Introduction setting context, Main sections with subsections building arguments, Conclusion synthesizing findings, References section).
- [ ] Logical hierarchy (e.g., balanced sections/subsections)
- [ ] Comprehensive coverage without disjointed jumps
- [ ] References section fully lists sources with URLs/details

### 3. CONTENT COMPREHENSIVENESS & QUALITY (Article Generation Stage)
- Score based on: Full-length report (2000-5000 words), neutral/encyclopedic tone, detailed analysis, and relevance to the topic. Content should reflect multi-hop reasoning (e.g., chaining perspectives).
- [ ] Appropriate length and depth (detailed, not verbose)
- [ ] Neutral, factual tone with analytical insights
- [ ] Multi-hop synthesis (e.g., connecting ideas across perspectives)

### 4. CITATION & GROUNDEDNESS (All Stages, Especially Research & Generation)
- Score based on: Proper inline citations [1] for all claims, verifiable sources (e.g., from web searches or pages), no fabrications, and accurate representation of sources.
- [ ] Inline citations for every key claim
- [ ] Citations link to real, diverse sources (no dead links or fakes inferred)
- [ ] Full groundedness (no extrapolations beyond cited info)

### 5. REFINEMENT & READABILITY (Polishing Stage)
- Score based on: Presence of executive summary, absence of duplicates/inconsistencies, coherent flow, high readability, and overall polish (e.g., grammar, clarity).
- [ ] Includes executive summary
- [ ] No redundancies or inconsistencies
- [ ] Excellent readability and flow

## SCORING & CLASSIFICATION RULES
- Compute total score: Average of the 5 criteria scores (0-10), rounded to one decimal place.
- Overall classification:
  - Total ≥ 9.0 → excellent (Fully aligns with STORM; publication-ready)
  - Total 7.0-8.9 → good (Strong but minor gaps in depth/structure)
  - Total 5.0-6.9 → fair (Adequate but lacks comprehensiveness/polish)
  - Total < 5.0 → poor (Significant deficiencies; does not meet STORM standards)
- Output in JSON format for parsability:
{
  "criteria_scores": {
    "research_depth": {"score": X, "justification": "Brief text"},
    "structure_organization": {"score": X, "justification": "Brief text"},
    "content_comprehensiveness": {"score": X, "justification": "Brief text"},
    "citation_groundedness": {"score": X, "justification": "Brief text"},
    "refinement_readability": {"score": X, "justification": "Brief text"}
  },
  "total_score": Y.Y,
  "overall_classification": "classification"
}


The problem/topic is provided below:
<problem>
{problem}
</problem>

The report to evaluate is provided below:
<report>
{report_content}
</report>

Please evaluate the report step by step, assigning scores and justifications, then output the JSON.
""".strip()


def interact(messages):

    match = re.search(
        r"<search>(.*?)</search>", messages[-1]["content"]
    )
    if match is None:
        return []
    
    query = match.group(1)
    result = requests.post(
        "http://localhost:8000/search", json={
            "query": query
        }
    ).json()

    return [
        {"role": "tool", "content": result}
    ]

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def reward_fn(messages, answer):
    """Original exact match reward function (deprecated - use llm_reward_fn instead)"""
    preds = re.findall(
        r"<answer>(.*?)</answer>", messages[-1]["content"]
    )
    if len(preds) == 0:
        return False
    pred = normalize_answer(preds[-1])

    if isinstance(answer, str):
        answer = [answer]
    answer = [normalize_answer(a) for a in answer]
  
    return pred in answer


# ============================================================================
# NEW REWARD FUNCTION IMPLEMENTATION
# Uses OpenAI client to call Gemini Flash directly, bypassing LangChain/RAGAS
# ============================================================================

# Global OpenAI client instance (initialized once for efficiency)
_openai_client = None

def get_openai_client():
    """Get or create the OpenAI client instance for Gemini Flash"""
    global _openai_client
    
    if _openai_client is None:
        from openai import OpenAI
        import os
        
        # Initialize OpenAI client for Gemini Flash
        _openai_client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY"),  # Use env var or fallback
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    return _openai_client


# Answer accuracy prompt templates adapted from RAGAS
TEMPLATE_ACCURACY_1 = (
    "Instruction: You are a world class state of the art assistant for rating "
    "a User Answer given a Question. The Question is completely answered by the Reference Answer.\n"
    "Say 4, if User Answer is full contained and equivalent to Reference Answer"
    "in all terms, topics, numbers, metrics, dates and units.\n"
    "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer"
    "in all terms, topics, numbers, metrics, dates and units.\n"
    "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics,"
    "numbers, metrics, dates and units or the User Answer do not answer the question.\n"
    "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.\n"
    "### Question: {query}\n"
    "### User Answer: {user_answer}\n"
    "### Reference Answer: {reference_answer}\n"
    "The rating is:\n"
)

TEMPLATE_ACCURACY_2 = (
    "I will rate the User Answer in comparison to the Reference Answer for a given Question.\n"
    "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.\n"
    "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.\n"
    "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.\n"
    "I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).\n"
    "Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.\n\n"
    "Question: {query}\n\n"
    "User Answer: {user_answer}\n\n"
    "Reference Answer: {reference_answer}\n\n"
    "Rating: "
)


def process_score(response_text: str) -> float:
    """Process the LLM response to extract a score between 0 and 1"""
    # Look for scores 0, 2, 4 in the response
    for score in [4, 2, 0]:
        if str(score) in response_text:
            return score / 4.0  # Normalize to 0-1 range
    return 0.0  # Default to 0 if no valid score found


def average_scores(score1: float, score2: float) -> float:
    """Average two scores, handling NaN values"""
    if score1 >= 0 and score2 >= 0:
        return (score1 + score2) / 2
    elif score1 >= 0:
        return score1
    elif score2 >= 0:
        return score2
    else:
        return 0.0


def llm_reward_fn(messages, answer: Union[str, List[str]], user_input: str = None) -> float:
    """
    LLM-based reward function using OpenAI client with Gemini Flash and answer accuracy scoring
    
    Args:
        messages: List of conversation messages
        answer: Ground truth answer(s) - string or list of strings
        user_input: The original question/prompt (if not provided, will try to extract from messages)
    
    Returns:
        float: Score between 0.0 and 1.0 representing answer accuracy
    """

    # Extract the model's response from messages
    preds = re.findall(
        r"<answer>(.*?)</answer>", messages[-1]["content"]
    )
    if len(preds) == 0:
        return 0.0
    
    response = preds[-1].strip()
    if not response:
        return 0.0
    
    # Prepare reference answer
    if isinstance(answer, str):
        reference = answer
    else:
        # If multiple reference answers, join them or use the first one
        reference = answer[0] if answer else ""
    
    # Extract user input if not provided
    if user_input is None:
        # Try to find the original question in the conversation
        for msg in messages:
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
        
        if not user_input:
            user_input = "Please evaluate the answer."
    
    # Get OpenAI client
    client = get_openai_client()
    
    # Score with first template
    prompt1 = TEMPLATE_ACCURACY_1.format(
        query=user_input,
        user_answer=response,
        reference_answer=reference
    )
    response1 = client.chat.completions.create(
        model="gemini-2.0-flash-lite",
        messages=[{"role": "user", "content": prompt1}],
        temperature=0.1,
        max_tokens=10,  # Short response expected
        timeout=30
    )
    score1 = process_score(response1.choices[0].message.content)
    
    # Score with second template
    prompt2 = TEMPLATE_ACCURACY_2.format(
        query=user_input,
        user_answer=response,
        reference_answer=reference
    )
    response2 = client.chat.completions.create(
        model="gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": prompt2}],
        temperature=0.1,
        max_tokens=10,  # Short response expected
        timeout=30
    )
    score2 = process_score(response2.choices[0].message.content)

    # Average the scores
    final_score = average_scores(score1, score2)
    
    return float(final_score)
