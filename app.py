import os
import requests
import json
import base64
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

# Flask imports
from flask import Flask, render_template, request, redirect, url_for

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# --- 1. CONFIGURATION FROM ENVIRONMENT ---

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Default to gemini-2.5-flash

# Validate required environment variables
if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, GEMINI_API_KEY]):
    raise ValueError(
        "Missing required environment variables. Please set SPOTIFY_CLIENT_ID, "
        "SPOTIFY_CLIENT_SECRET, and GEMINI_API_KEY in your .env file."
    )

# --- GLOBAL SPOTIFY TOKEN ---
SPOTIFY_ACCESS_TOKEN = None

# --- SPOTIFY AUTHENTICATION ---

def get_spotify_token(client_id, client_secret):
    """Obtains a Spotify Bearer Token."""
    auth_string = f"{client_id}:{client_secret}"
    auth_header = base64.b64encode(auth_string.encode()).decode()
    token_url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    try:
        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()
        token_info = response.json()
        return token_info.get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not obtain Spotify token. Error: {e}")
        return None

# --- 2. SPOTIFY TOOL DEFINITION ---

@tool
def search_spotify(query: str, search_type: str = "episode", limit: int = 20) -> str:
    """
    Searches Spotify for podcast episodes.
    
    Args:
        query: The search query string optimized for Spotify's search API
        search_type: Must be 'episode' (default) for podcast episodes
        limit: Number of results to return (default 20, max 50)
    
    Returns a JSON string containing episode information.
    """
    global SPOTIFY_ACCESS_TOKEN
    
    if not SPOTIFY_ACCESS_TOKEN:
        return json.dumps({"error": "Spotify token not available. Cannot search."})

    # Ensure we're searching for episodes
    if search_type != "episode":
        search_type = "episode"

    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {SPOTIFY_ACCESS_TOKEN}"}
    params = {
        "q": query,
        "type": search_type,
        "limit": min(limit, 50)  # Spotify API limit
    }

    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        episodes = []
        items = data.get('episodes', {}).get('items', [])
        for item in items:
            # Extract images - Spotify provides images array, use the largest one
            images = item.get('images', [])
            image_url = images[0].get('url') if images else None
            
            # Also get show image as fallback
            show_images = item.get('show', {}).get('images', [])
            show_image_url = show_images[0].get('url') if show_images else None
            
            episodes.append({
                "episode_name": item.get('name'),
                "show_name": item.get('show', {}).get('name'),
                "release_date": item.get('release_date'),
                "description": item.get('description', ''),
                "external_urls": item.get('external_urls', {}).get('spotify'),
                "duration_ms": item.get('duration_ms', 0),
                "image_url": image_url or show_image_url,  # Use episode image or show image as fallback
                "show_id": item.get('show', {}).get('id')
            })

        return json.dumps(episodes, indent=2)

    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Spotify search failed: {e}"})

# --- 3. LANGGRAPH STATE & MODEL SETUP ---

class AgentState(TypedDict):
    """Represents the state of the agent's workflow."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    original_query: str
    optimized_query: Optional[str]
    search_limit: int
    raw_episodes: List[dict]
    selected_episodes: List[dict]
    summary: Optional[str]

# Initialize the Gemini Model (lazy initialization)
llm = None
llm_with_tools = None
tools = [search_spotify]

def get_llm():
    """Get the base LLM."""
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.0
        )
    return llm

def get_llm_with_tools():
    """Get LLM with tools bound."""
    global llm_with_tools
    if llm_with_tools is None:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

# --- 4. GRAPH NODES ---

def query_reasoning_node(state: AgentState):
    """
    AI reasons about the user's natural language query and creates an optimized Spotify search query.
    Uses first principles: intent extraction, temporal reasoning, search optimization.
    """
    original_query = state.get("original_query", "")
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    llm = get_llm()
    
    reasoning_prompt = f"""You are an expert at understanding user intent and optimizing search queries. Your task is to analyze a natural language query and create the most effective search query for finding podcast episodes on Spotify.

CURRENT CONTEXT:
- Today's date: {current_date.strftime('%B %d, %Y')}
- Current year: {current_year}
- Current month: {current_month}

USER'S ORIGINAL QUERY: "{original_query}"

YOUR ANALYSIS PROCESS:

1. INTENT EXTRACTION:
   - What is the user truly seeking? (topic, theme, specific information, learning goal)
   - Are there implicit requirements? (depth level, audience, perspective, format preference)
   - What would make an episode "relevant" for this query?
   - Is the query ambiguous? If so, identify the most likely interpretation based on common podcast topics
   - Does the query have multiple facets? (e.g., "AI for beginners" = AI + beginner level)

2. USER INTENT DEPTH ANALYSIS:
   - Learning level: beginner, intermediate, advanced, expert (infer from words like "learn", "basics", "deep dive", "advanced")
   - Purpose: education, entertainment, news/current events, research, practical application
   - Scope: broad overview vs. specific subtopic vs. deep dive
   - Context: standalone episode vs. series/continuation

3. TEMPORAL REASONING:
   - Does the query express temporal preferences? (recent, latest, new, current, up-to-date, historical, classic, timeless)
   - If "recent" is mentioned: episodes from the last 6-12 months are recent relative to {current_date.strftime('%B %d, %Y')}
   - If no temporal preference: don't add year constraints (cast a wider net)
   - Calculate what "recent" means: Today is {current_date.strftime('%B %d, %Y')}, so recent would be approximately {current_year} or late {current_year - 1}
   - Consider topic temporal sensitivity:
     * Time-sensitive: current events, technology, recent research, news → recency matters
     * Time-agnostic: philosophy, history, fundamentals, timeless concepts → recency less important
   - Historical queries: If user asks for "historical" or "classic", prioritize older episodes

4. SEMANTIC UNDERSTANDING:
   - Identify synonyms and related terms (e.g., "AI" = "artificial intelligence" = "machine learning" in some contexts)
   - Consider broader and narrower interpretations
   - Recognize domain-specific terminology
   - Handle acronyms and abbreviations appropriately

5. KEYWORD EXTRACTION:
   - Extract core concepts, topics, and entities
   - Remove conversational filler ("I want", "please", "can you", "show me")
   - Preserve important qualifiers (beginner, advanced, specific techniques, formats)
   - If query mentions "podcasts" or "episodes", remove these words (they're implicit)
   - Handle negations carefully (e.g., "not about X" - exclude X from search)

6. SEARCH OPTIMIZATION:
   - Spotify search works best with: core topic keywords, specific terms, proper nouns
   - Combine related concepts intelligently (use AND logic implicitly)
   - If temporal preference exists, consider adding year only if it improves search results
   - Keep query concise but comprehensive (3-8 keywords typically optimal)
   - For very specific queries, use exact terminology
   - For broad queries, use general terms to cast wider net
   - Consider alternative phrasings if primary terms might not yield results

7. QUERY AMBIGUITY RESOLUTION:
   - If query is vague (e.g., "good podcasts"), infer intent from context or use broad search
   - If query has multiple interpretations, choose the most common podcast-related interpretation
   - For ambiguous terms, include both interpretations in search if possible

8. RESULT SCOPE:
   - Determine appropriate result limit (default 50, which is Spotify's maximum)
   - For broad queries: use 50 to give AI more options to choose from
   - For very specific queries: can use 30-40, but 50 is fine since AI will filter
   - More results = better filtering decisions by AI

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no code blocks, no explanations outside JSON):

{{
    "optimized_query": "keyword1 keyword2 keyword3 [year if temporal preference]",
    "reasoning": "Brief explanation: extracted intent, temporal handling, keyword choices",
    "temporal_preference": "recent|any|historical",
    "limit": 50
}}

EXAMPLE:
Query: "recent podcasts about climate change"
Output: {{"optimized_query": "climate change {current_year}", "reasoning": "Extracted core topic 'climate change'. Added {current_year} for recency as user requested recent content.", "temporal_preference": "recent", "limit": 50}}

Now analyze the user's query and return the JSON response.
"""

    result = llm.invoke([HumanMessage(content=reasoning_prompt)])
    content = result.content
    
    # Parse the JSON response
    try:
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        query_data = json.loads(content)
        optimized_query = query_data.get("optimized_query", original_query)
        limit = query_data.get("limit", 50)  # Default to 50 (Spotify's max)
        
        return {
            "optimized_query": optimized_query,
            "search_limit": limit,
            "messages": [AIMessage(content=f"Optimized query: {optimized_query}")]
        }
    except json.JSONDecodeError:
        # Fallback: use original query
        return {
            "optimized_query": original_query,
            "search_limit": 50,  # Default to 50 (Spotify's max)
            "messages": [AIMessage(content=f"Using original query: {original_query}")]
        }

def search_node(state: AgentState):
    """Executes the Spotify search with the optimized query."""
    optimized_query = state.get("optimized_query", "")
    limit = state.get("search_limit", 50)  # Default to 50 (Spotify's max)
    
    # Call the search tool
    tool_output = search_spotify.invoke({
        "query": optimized_query,
        "search_type": "episode",
        "limit": limit
    })
    
    # Parse episodes
    try:
        episodes = json.loads(tool_output)
        if not isinstance(episodes, list):
            episodes = []
    except json.JSONDecodeError:
        episodes = []
    
    return {
        "raw_episodes": episodes,
        "messages": [ToolMessage(content=tool_output, tool_call_id="search_spotify")]
    }

def result_filtering_node(state: AgentState):
    """
    AI reasons about the original query and search results using first principles.
    Evaluates relevance, temporal alignment, and quality to select optimal episodes.
    """
    original_query = state.get("original_query", "")
    raw_episodes = state.get("raw_episodes", [])
    current_date = datetime.now()
    current_year = current_date.year
    
    if not raw_episodes:
        return {
            "selected_episodes": [],
            "summary": "No episodes found matching your query.",
            "messages": [AIMessage(content="No episodes found.")]
        }
    
    llm = get_llm()
    
    # Prepare episode data for the LLM (Gemini has 1M context window, so we can include all episodes)
    episodes_json = json.dumps(raw_episodes, indent=2)
    
    # Calculate date ranges for temporal reasoning
    six_months_ago = current_date - relativedelta(months=6)
    one_year_ago = current_date - relativedelta(years=1)
    
    filtering_prompt = f"""You are an expert at evaluating podcast episode relevance and quality. Your task is to analyze search results and select the episodes that best match the user's original intent.

CURRENT CONTEXT:
- Today's date: {current_date.strftime('%B %d, %Y')}
- Current year: {current_year}
- Episodes from the last 6 months (since {six_months_ago.strftime('%B %d, %Y')}) are considered "recent"
- Episodes from the last year (since {one_year_ago.strftime('%B %d, %Y')}) are considered "relatively recent"

USER'S ORIGINAL QUERY: "{original_query}"

AVAILABLE EPISODES (from Spotify search):
{episodes_json}

EVALUATION FRAMEWORK:

1. INTENT ALIGNMENT:
   - Does the episode address the core topic/theme the user is seeking?
   - How directly does it relate to the user's stated interest? (direct match > tangential > related)
   - Does it match the inferred learning level? (beginner-friendly vs. advanced)
   - Does it match the inferred purpose? (educational vs. entertainment vs. news)
   - Are there multiple episodes on the same topic? Prioritize variety and depth
   - Score: 1.0 (perfect match) to 0.0 (unrelated)

2. TEMPORAL RELEVANCE:
   - Did the user express temporal preferences? (recent, latest, current, new, historical, classic)
   - If "recent" was mentioned: prioritize episodes from {current_year} or late {current_year - 1}
   - If no temporal preference: consider all episodes, but newer may be slightly preferred if topic is time-sensitive
   - For time-sensitive topics (current events, technology, recent research), recency matters more
   - For timeless topics (philosophy, history, fundamentals), recency matters less
   - Historical queries: If user asked for "historical" or "classic", older episodes may be preferred
   - Calculate temporal score based on release date relative to user's preference

3. CONTENT QUALITY INDICATORS:
   - Episode title clarity and specificity (clear titles > vague titles)
   - Description completeness and relevance (detailed descriptions > sparse descriptions)
   - Show/podcast reputation (if recognizable - established shows often have higher quality)
   - Episode length (if available) - longer episodes often have more depth, but not always
   - Description quality: Does it explain what the episode covers? Is it informative?
   - Title quality: Does it clearly indicate the episode's focus?

4. DIVERSITY AND COMPREHENSIVENESS:
   - Select episodes that offer different perspectives or aspects of the topic
   - Avoid selecting multiple episodes from the same show unless they offer distinct value
   - Aim for a balanced representation of the topic (different angles, different experts, different formats)
   - If all good results are from one show, that's acceptable, but prefer variety when possible
   - Consider format diversity: interviews, solo episodes, panel discussions, etc.

5. RESULT SET QUALITY ASSESSMENT:
   - Evaluate the overall quality of available results
   - If results are poor (low relevance, low quality), be more selective
   - If results are excellent (high relevance, high quality), you can be more selective
   - If very few results exist, include all relevant ones even if quality varies
   - If many excellent results exist, prioritize the absolute best matches

6. EDGE CASE HANDLING:
   - If all results are from the same show: Accept this if they're the best matches, but note it in reasoning
   - If results are too old (for recent queries): Only include if they're exceptionally relevant
   - If results are too new (for historical queries): Only include if they're about historical topics
   - If very few results: Include all relevant ones, even if quality varies
   - If too many results: Be more selective, focus on best matches

7. SEMANTIC RELEVANCE:
   - Does the episode match the semantic meaning, not just keywords?
   - Consider synonyms and related concepts (e.g., "AI" matches "artificial intelligence", "machine learning")
   - Avoid false positives: episodes that mention keywords but aren't actually about the topic
   - Prefer episodes where the topic is central, not just mentioned in passing

8. RANKING CRITERIA (in order of importance):
   a) Direct relevance to user's core intent (must be on-topic)
   b) Temporal alignment (if user specified recency preference)
   c) Content quality and depth indicators
   d) Diversity of perspectives and sources
   e) Semantic match quality (central topic vs. tangential mention)

SELECTION GUIDELINES:
- Select 5-10 episodes that best match the user's intent (aim for 7-8 for optimal balance)
- Prioritize episodes that directly address the query over tangentially related ones
- If user asked for "recent" content, ensure selected episodes are from {current_year} or late {current_year - 1} when possible
- If temporal preference wasn't specified, don't artificially limit to recent episodes
- Order by relevance: most directly relevant first
- If fewer than 5 highly relevant episodes exist, include all relevant ones (even if < 5)
- If more than 10 highly relevant episodes exist, select the top 10 by combined score
- Balance quality with diversity: prefer variety unless one source is clearly superior

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no code blocks, no explanations outside JSON):

{{
    "selected_episode_names": ["exact episode name 1", "exact episode name 2", ...],
    "summary": "A clear, concise explanation (2-4 sentences) that: (1) explains why these episodes were selected, (2) how they relate to the user's query, (3) what makes them relevant, (4) addresses temporal reasoning if applicable, (5) notes any diversity considerations (e.g., 'from multiple shows' or 'all from X show but covering different aspects'). Be specific and informative."
}}

CRITICAL REQUIREMENTS:
- Return EXACT episode names as they appear in the "episode_name" field (case-sensitive, character-for-character match)
- Order episodes by relevance (most relevant first)
- If user asked for "recent" content, ensure your reasoning reflects proper temporal understanding relative to {current_date.strftime('%B %d, %Y')}
- Be precise in your summary about why each selection was made
- If results are limited or all from one source, acknowledge this in the summary
- If you had to make trade-offs (e.g., quality vs. recency), explain the reasoning
- Never include episodes that don't match the core topic, even if they're recent or high-quality

Now evaluate the episodes and return the JSON response.
"""

    result = llm.invoke([HumanMessage(content=filtering_prompt)])
    content = result.content
    
    # Parse the JSON response
    try:
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        filter_data = json.loads(content)
        selected_names = filter_data.get("selected_episode_names", [])
        summary = filter_data.get("summary", "Selected relevant episodes based on your query.")
        
        # Match selected names to full episode data
        selected_episodes = []
        for name in selected_names:
            # Try exact match first
            matching = next(
                (e for e in raw_episodes if e.get("episode_name") == name),
                None
            )
            # Try partial match if exact fails
            if not matching:
                matching = next(
                    (e for e in raw_episodes if name.lower() in e.get("episode_name", "").lower()),
                    None
                )
            if matching and matching not in selected_episodes:
                selected_episodes.append(matching)
        
        # If no matches found, use top 5 as fallback
        if not selected_episodes:
            selected_episodes = raw_episodes[:5]
        
        return {
            "selected_episodes": selected_episodes,
            "summary": summary,
            "messages": [AIMessage(content=summary)]
        }
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback: return top 5 episodes
        return {
            "selected_episodes": raw_episodes[:5],
            "summary": f"Found {len(raw_episodes)} episodes related to '{original_query}'. Here are the top results.",
            "messages": [AIMessage(content=f"Found {len(raw_episodes)} episodes.")]
        }

# --- 5. GRAPH CONSTRUCTION ---

def build_graph():
    """Constructs and compiles the LangGraph state machine."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("query_reasoning", query_reasoning_node)
    workflow.add_node("search", search_node)
    workflow.add_node("result_filtering", result_filtering_node)

    # Set entry point
    workflow.set_entry_point("query_reasoning")

    # Define edges (linear flow)
    workflow.add_edge("query_reasoning", "search")
    workflow.add_edge("search", "result_filtering")
    workflow.add_edge("result_filtering", END)

    # Compile the graph
    return workflow.compile()

# --- 6. FLASK ROUTES ---

def search_podcasts(query: str):
    """Search for podcast episodes using the AI-native workflow."""
    global SPOTIFY_ACCESS_TOKEN
    
    # Ensure we have a Spotify token
    if not SPOTIFY_ACCESS_TOKEN:
        SPOTIFY_ACCESS_TOKEN = get_spotify_token(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
        if not SPOTIFY_ACCESS_TOKEN:
            return None, "Could not obtain Spotify access token. Please check your API keys."

    # Build the Agent
    agent_app = build_graph()

    # Define the initial state
    initial_state = {
        "messages": [],
        "original_query": query,
        "optimized_query": None,
        "search_limit": 50,  # Default to 50 (Spotify's max)
        "raw_episodes": [],
        "selected_episodes": [],
        "summary": None
    }

    # Run the Agent
    final_state = None
    for step in agent_app.stream(initial_state):
        # Get the final state from result_filtering node or END
        if "result_filtering" in step:
            final_state = step["result_filtering"]
        elif END in step:
            final_state = step[END]

    if final_state:
        episodes = final_state.get("selected_episodes", [])
        summary = final_state.get("summary", "Found relevant episodes.")
        return episodes, summary
    
    return [], "No results found."

# Jinja2 filter for formatting duration
@app.template_filter('format_duration')
def format_duration(duration_ms):
    """Format duration in milliseconds to readable format."""
    if not duration_ms:
        return None
    minutes = int(duration_ms / 60000)
    hours = minutes // 60
    minutes = minutes % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

@app.route('/')
def index():
    """Home page with search form."""
    global SPOTIFY_ACCESS_TOKEN
    
    # Check API status
    spotify_status = SPOTIFY_ACCESS_TOKEN is not None
    if not spotify_status:
        # Try to get token to verify credentials
        test_token = get_spotify_token(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
        spotify_status = test_token is not None
        if test_token:
            SPOTIFY_ACCESS_TOKEN = test_token
    
    gemini_status = GEMINI_API_KEY is not None and len(GEMINI_API_KEY.strip()) > 0
    
    return render_template('index.html', 
                         spotify_status=spotify_status,
                         gemini_status=gemini_status)

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Handle search requests."""
    # If GET request (direct visit), redirect to homepage
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    # Handle POST request (form submission)
    query = request.form.get('query', '').strip()
    
    if not query:
        return render_template('index.html', error="Please enter a search query.")
    
    try:
        episodes, summary = search_podcasts(query)
        
        if episodes is None:
            return render_template('index.html', error=summary)
        
        return render_template('results.html', query=query, episodes=episodes, summary=summary)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

# --- 7. INITIALIZATION AND MAIN EXECUTION ---

if __name__ == "__main__":
    # Initialize Spotify Token on startup
    SPOTIFY_ACCESS_TOKEN = get_spotify_token(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    
    if not SPOTIFY_ACCESS_TOKEN:
        print("WARNING: Could not obtain Spotify token on startup. Will retry on first request.")
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5002)
