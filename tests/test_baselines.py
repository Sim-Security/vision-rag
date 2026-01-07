"""Test baseline: Known question-answer pairs for Vision RAG validation.

Each test case includes:
- image: Path to test image (relative to tests/images/)
- query: Test question
- expected_keywords: Keywords that MUST appear in a correct answer
- expected_topic: General topic the answer should address
- difficulty: easy/medium/hard for self-correction testing
"""

TEST_CASES = [
    # ==========================================================================
    # FINANCIAL CHARTS - Easy (high relevance, clear answers)
    # NOTE: expected_keywords should include SPECIFIC VALUES from the image
    # to catch hallucinations where keywords match but numbers are wrong
    # ==========================================================================
    {
        "id": "fin_001",
        "category": "charts",
        "image": "charts/nike_financials.png",
        "query": "What is Nike's net profit?",
        # Image shows: Net profit $0.8B, 7% margin
        "expected_keywords": ["0.8", "profit", "nike"],
        "expected_topic": "Nike's financial performance and net profit figures",
        "difficulty": "easy",
        "ground_truth": "$0.8B net profit (7% margin)",
    },
    {
        "id": "fin_002", 
        "category": "charts",
        "image": "charts/tesla_financials.png",
        "query": "What is Tesla's revenue?",
        # Image shows: Revenue $19.3B (9% Y/Y), Net profit $0.4B
        "expected_keywords": ["19.3", "revenue", "tesla"],
        "expected_topic": "Tesla's Q1 FY25 revenue",
        "difficulty": "easy",
        "ground_truth": "Revenue $19.3B (9% Y/Y), Net profit $0.4B (2% margin)",
    },
    {
        "id": "fin_003",
        "category": "charts",
        "image": "charts/netflix_financials.png",
        # NOTE: Image is income statement, NOT subscriber data
        "query": "What is Netflix's net profit?",
        # Image shows: Revenue $10.5B, Net profit $2.9B (27% margin)
        "expected_keywords": ["2.9", "profit", "netflix"],
        "expected_topic": "Netflix Q1 FY25 net profit",
        "difficulty": "easy",
        "ground_truth": "Net profit $2.9B (27% margin), Revenue $10.5B",
    },
    {
        "id": "fin_004",
        "category": "charts",
        "image": "charts/google_financials.png",
        # NOTE: Image is about ACQUISITIONS, not advertising revenue!
        "query": "What was Google's biggest acquisition?",
        # Image shows: Wiz $32.0B (2026), Motorola $12.5B, Mandiant $5.4B
        "expected_keywords": ["wiz", "32", "billion"],
        "expected_topic": "Google's largest acquisition (Wiz)",
        "difficulty": "easy",
        "ground_truth": "Wiz at $32.0B (2026 expected), Motorola $12.5B (2012)",
    },
    {
        "id": "fin_005",
        "category": "charts",
        "image": "charts/accenture_financials.png",
        # NOTE: Image shows GenAI Bookings, not general revenue
        "query": "What are Accenture's Generative AI bookings?",
        # Image shows: Q2 FY25 = $1.4B, growing from $0.1B in Q3 FY23
        "expected_keywords": ["1.4", "generative", "ai"],
        "expected_topic": "Accenture GenAI bookings trend",
        "difficulty": "easy",
        "ground_truth": "Q2 FY25: $1.4B in GenAI bookings",
    },
    {
        "id": "fin_006",
        "category": "charts",
        "image": "charts/tencent_financials.png",
        "query": "What is Tencent's 2024 revenue?",
        # Image shows: 660B RMB ($92B) in 2024, segments: Gaming, Social, Marketing, Finance
        "expected_keywords": ["660", "92"],
        "expected_topic": "Tencent 2024 revenue",
        "difficulty": "easy",
        "ground_truth": "660B RMB (~$92B USD) in 2024",
    },
    
    # ==========================================================================
    # COMPARATIVE QUERIES - Medium (requires selecting correct image)
    # NOTE: These test retrieval quality - no specific image expected
    # ==========================================================================
    {
        "id": "comp_001",
        "category": "comparative",
        "image": None,
        "query": "Which company has the highest net profit margin?",
        # Netflix has 27% margin (highest), Nike 7%, Tesla 2%
        "expected_keywords": ["netflix", "27", "margin"],
        "expected_topic": "Profit margin comparison",
        "difficulty": "medium",
        "test_type": "retrieval",
        "ground_truth": "Netflix at 27% margin is highest",
    },
    {
        "id": "comp_002",
        "category": "comparative",
        "image": None,
        "query": "Show me streaming service financials",
        # Should retrieve Netflix income statement
        "expected_keywords": ["netflix", "10.5", "revenue"],
        "expected_topic": "Should retrieve Netflix",
        "difficulty": "medium",
        "test_type": "retrieval",
        "ground_truth": "Netflix Q1 FY25: $10.5B revenue, $2.9B profit",
    },
    {
        "id": "comp_003",
        "category": "comparative",
        "image": None,
        "query": "What tech company in China has financial data?",
        "expected_keywords": ["tencent", "660"],
        "expected_topic": "Should retrieve Tencent",
        "difficulty": "medium",
        "test_type": "retrieval",
        "ground_truth": "Tencent: 660B RMB revenue in 2024",
    },
    
    # ==========================================================================
    # SPECIFIC QUERIES - Hard (requires precise reading)
    # ==========================================================================
    {
        "id": "spec_001",
        "category": "specific",
        "image": "charts/tesla_financials.png",
        "query": "What is Tesla's net profit margin?",
        # Image shows: Net profit $0.4B, 2% margin
        "expected_keywords": ["2", "margin", "profit"],
        "expected_topic": "Tesla profit margin",
        "difficulty": "hard",
        "ground_truth": "2% net profit margin",
    },
    {
        "id": "spec_002",
        "category": "specific",
        "image": "charts/nike_financials.png",
        "query": "What is Nike's gross margin percentage?",
        # Image shows: Gross profit $4.7B, 41% margin
        "expected_keywords": ["41", "margin", "gross"],
        "expected_topic": "Nike's gross margin",
        "difficulty": "hard",
        "ground_truth": "41% gross margin",
    },
    
    # ==========================================================================
    # NEGATIVE CASES - Should trigger "no relevant image" response
    # ==========================================================================
    {
        "id": "neg_001",
        "category": "negative",
        "image": None,
        "query": "What is the recipe for chocolate cake?",
        "expected_keywords": [],
        "expected_topic": None,
        "difficulty": "hard",
        "test_type": "no_match",  # Should return "no relevant image"
    },
    {
        "id": "neg_002",
        "category": "negative",
        "image": None,
        "query": "Explain quantum mechanics",
        "expected_keywords": [],
        "expected_topic": None,
        "difficulty": "hard",
        "test_type": "no_match",
    },
    
    # ==========================================================================
    # TECHNICAL DIAGRAMS - Tests visual understanding
    # Generated images with known values
    # ==========================================================================
    {
        "id": "diag_001",
        "category": "diagrams",
        "image": "diagrams/neural_network.png",
        "query": "How many layers does this neural network have?",
        "expected_keywords": ["3", "layer"],
        "expected_topic": "Neural network layer count",
        "difficulty": "medium",
        "ground_truth": "3 layers: input (4 nodes), hidden (5 nodes), output (2 nodes)",
    },
    {
        "id": "diag_002",
        "category": "diagrams",
        "image": "diagrams/neural_network.png",
        "query": "How many nodes are in the hidden layer?",
        "expected_keywords": ["5", "hidden"],
        "expected_topic": "Hidden layer node count",
        "difficulty": "medium",
        "ground_truth": "5 nodes in hidden layer",
    },
    {
        "id": "diag_003",
        "category": "diagrams",
        "image": "diagrams/flowchart_process.png",
        "query": "What happens when a support ticket is urgent?",
        "expected_keywords": ["escalate", "manager"],
        "expected_topic": "Urgent ticket handling process",
        "difficulty": "easy",
        "ground_truth": "Escalate to Manager",
    },
    {
        "id": "diag_004",
        "category": "diagrams",
        "image": "diagrams/flowchart_process.png",
        "query": "What are the steps in this customer support process?",
        "expected_keywords": ["ticket", "agent", "resolve"],
        "expected_topic": "Customer support workflow",
        "difficulty": "medium",
        "ground_truth": "Receive Ticket → Is Urgent? → Assign to Agent → Resolve → Close",
    },
    
    # ==========================================================================
    # GENERIC BUSINESS CHARTS - Generated images with known values
    # ==========================================================================
    {
        "id": "chart_001",
        "category": "charts_generic",
        "image": "charts_generic/sales_bar_chart.png",
        "query": "What were Q4 sales?",
        "expected_keywords": ["4.2", "Q4"],
        "expected_topic": "Q4 sales figure",
        "difficulty": "easy",
        "ground_truth": "Q4: $4.2M",
    },
    {
        "id": "chart_002",
        "category": "charts_generic",
        "image": "charts_generic/sales_bar_chart.png",
        "query": "Which quarter had the lowest sales?",
        "expected_keywords": ["Q1", "2.3"],
        "expected_topic": "Lowest sales quarter",
        "difficulty": "medium",
        "ground_truth": "Q1: $2.3M (lowest)",
    },
    {
        "id": "chart_003",
        "category": "charts_generic",
        "image": "charts_generic/pie_chart_budget.png",
        "query": "What percentage of the budget goes to Engineering?",
        "expected_keywords": ["40", "engineering"],
        "expected_topic": "Engineering budget percentage",
        "difficulty": "easy",
        "ground_truth": "Engineering: 40%",
    },
    {
        "id": "chart_004",
        "category": "charts_generic",
        "image": "charts_generic/pie_chart_budget.png",
        "query": "Which department has the smallest budget allocation?",
        "expected_keywords": ["HR", "10"],
        "expected_topic": "Smallest budget department",
        "difficulty": "medium",
        "ground_truth": "HR: 10% (smallest)",
    },
    {
        "id": "chart_005",
        "category": "charts_generic",
        "image": "charts_generic/line_graph_growth.png",
        "query": "What was the company's revenue in 2024?",
        "expected_keywords": ["45", "revenue"],
        "expected_topic": "2024 revenue figure",
        "difficulty": "easy",
        "ground_truth": "2024 Revenue: $45M",
    },
    {
        "id": "chart_006",
        "category": "charts_generic",
        "image": "charts_generic/line_graph_growth.png",
        "query": "How much did profit grow from 2019 to 2024?",
        "expected_keywords": ["2", "12", "profit"],
        "expected_topic": "Profit growth over time",
        "difficulty": "medium",
        "ground_truth": "Profit: $2M (2019) → $12M (2024)",
    },

    # ==========================================================================
    # INFOGRAPHICS - Generated images with known values
    # ==========================================================================
    {
        "id": "info_001",
        "category": "infographics",
        "image": "infographics/world_map_sales.png",
        "query": "Which region has the highest sales?",
        "expected_keywords": ["north america", "50"],
        "expected_topic": "Highest sales region",
        "difficulty": "easy",
        "ground_truth": "North America: $50M",
    },
    {
        "id": "info_002",
        "category": "infographics",
        "image": "infographics/world_map_sales.png",
        "query": "What are the sales in Asia-Pacific?",
        "expected_keywords": ["28", "asia"],
        "expected_topic": "Asia-Pacific sales",
        "difficulty": "easy",
        "ground_truth": "Asia-Pacific: $28M",
    },
    {
        "id": "info_003",
        "category": "infographics",
        "image": "infographics/world_map_sales.png",
        "query": "What is the total global sales across all regions?",
        "expected_keywords": ["125"],
        "expected_topic": "Total global sales calculation",
        "difficulty": "hard",
        "ground_truth": "$50M + $35M + $28M + $12M = $125M total",
    },
]


def get_test_cases_by_difficulty(difficulty: str) -> list[dict]:
    """Filter test cases by difficulty level."""
    return [tc for tc in TEST_CASES if tc.get("difficulty") == difficulty]


def get_test_cases_by_category(category: str) -> list[dict]:
    """Filter test cases by category."""
    return [tc for tc in TEST_CASES if tc.get("category") == category]


def get_retrieval_tests() -> list[dict]:
    """Get tests specifically for retrieval quality."""
    return [tc for tc in TEST_CASES if tc.get("test_type") == "retrieval"]


def get_negative_tests() -> list[dict]:
    """Get tests that should return 'no relevant image'."""
    return [tc for tc in TEST_CASES if tc.get("test_type") == "no_match"]
