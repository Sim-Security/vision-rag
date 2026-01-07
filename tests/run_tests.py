"""Test runner for Vision RAG - validates retrieval and answer quality."""
import os
import sys
import json
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.config import AVAILABLE_MODELS
from src.state import VisionRAGState, GraphConfig
from src.tools import (
    get_cohere_client,
    get_chromadb_client,
    get_or_create_collection,
    embed_images,
    add_images_to_chromadb,
    get_collection_count,
    clear_collection,
)
from src.graph import create_vision_rag_graph

from test_baselines import TEST_CASES, get_test_cases_by_difficulty
from download_test_images import download_test_images


class TestResult:
    """Result of a single test case."""
    def __init__(self, test_id: str, query: str):
        self.test_id = test_id
        self.query = query
        self.passed = False
        self.retrieved_image: Optional[str] = None
        self.expected_image: Optional[str] = None
        self.answer: str = ""
        self.expected_keywords: list[str] = []
        self.found_keywords: list[str] = []
        self.missing_keywords: list[str] = []
        self.is_relevant: bool = False
        self.iterations: int = 0
        self.error: Optional[str] = None
        self.duration_ms: float = 0
    
    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "query": self.query,
            "passed": self.passed,
            "retrieved_image": self.retrieved_image,
            "expected_image": self.expected_image,
            "answer_preview": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "found_keywords": self.found_keywords,
            "missing_keywords": self.missing_keywords,
            "is_relevant": self.is_relevant,
            "iterations": self.iterations,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


def check_keywords(answer: str, expected_keywords: list[str]) -> tuple[list[str], list[str]]:
    """Check which expected keywords are in the answer.
    
    Returns:
        (found_keywords, missing_keywords)
    """
    answer_lower = answer.lower()
    found = []
    missing = []
    
    for kw in expected_keywords:
        if kw.lower() in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)
    
    return found, missing


def run_single_test(
    test_case: dict,
    config: GraphConfig,
    verbose: bool = True,
) -> TestResult:
    """Run a single test case and return the result."""
    import time
    
    result = TestResult(test_case["id"], test_case["query"])
    result.expected_image = test_case.get("image")
    result.expected_keywords = test_case.get("expected_keywords", [])
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test: {test_case['id']} ({test_case.get('difficulty', 'unknown')})")
        print(f"Query: {test_case['query']}")
    
    start_time = time.time()
    
    try:
        # Create initial state
        initial_state = VisionRAGState(
            query=test_case["query"],
            max_iterations=2,
        )
        
        # Create and run graph
        graph = create_vision_rag_graph(config)
        final_state_dict = graph.invoke(initial_state.model_dump())
        final_state = VisionRAGState(**final_state_dict)
        
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Extract results
        result.answer = final_state.answer
        result.is_relevant = final_state.is_relevant
        result.iterations = final_state.iteration
        
        if final_state.retrieved_images and final_state.current_image_idx < len(final_state.retrieved_images):
            result.retrieved_image = final_state.retrieved_images[final_state.current_image_idx].path
        
        # Check keywords
        result.found_keywords, result.missing_keywords = check_keywords(
            result.answer, result.expected_keywords
        )
        
        # Determine pass/fail
        test_type = test_case.get("test_type", "qa")
        
        if test_type == "no_match":
            # Negative test - should NOT find relevant image
            result.passed = not result.is_relevant or "no relevant" in result.answer.lower()
        elif test_type == "retrieval":
            # Just needs to retrieve something relevant
            result.passed = result.is_relevant and len(result.found_keywords) > 0
        else:
            # Standard QA - needs keywords and relevance
            keyword_ratio = len(result.found_keywords) / len(result.expected_keywords) if result.expected_keywords else 1
            result.passed = result.is_relevant and keyword_ratio >= 0.5
        
        if verbose:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"Status: {status}")
            print(f"Retrieved: {result.retrieved_image}")
            print(f"Relevant: {result.is_relevant}")
            print(f"Keywords: {len(result.found_keywords)}/{len(result.expected_keywords)}")
            if result.missing_keywords:
                print(f"Missing: {result.missing_keywords}")
            print(f"Duration: {result.duration_ms:.0f}ms")
        
    except Exception as e:
        result.error = str(e)
        result.duration_ms = (time.time() - start_time) * 1000
        if verbose:
            print(f"❌ ERROR: {e}")
    
    return result


def run_test_suite(
    test_cases: Optional[list[dict]] = None,
    model_key: str = "gemini-2.5-flash",
    verbose: bool = True,
    save_results: bool = True,
) -> dict:
    """Run the full test suite.
    
    Args:
        test_cases: Optional list of test cases (defaults to all)
        model_key: Which model to use from config
        verbose: Print progress
        save_results: Save JSON results file
        
    Returns:
        Summary dict with pass/fail counts
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    # Get API keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    
    if not openrouter_key or not cohere_key:
        print("❌ Missing API keys. Set OPENROUTER_API_KEY and COHERE_API_KEY in .env")
        return {"error": "Missing API keys"}
    
    # Create config
    model_id = AVAILABLE_MODELS.get(model_key, {}).get("id", "google/gemini-2.5-flash-preview")
    config = GraphConfig(
        model_id=model_id,
        human_in_loop=False,
        top_k=3,
        openrouter_api_key=openrouter_key,
        cohere_api_key=cohere_key,
    )
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"VISION RAG TEST SUITE")
        print(f"Model: {model_key} ({model_id})")
        print(f"Test cases: {len(test_cases)}")
        print(f"{'#'*60}")
    
    # Check ChromaDB has images
    try:
        client = get_chromadb_client()
        collection = get_or_create_collection(client)
        count = get_collection_count(collection)
        if count == 0:
            print("\n⚠️  ChromaDB is empty! Run index_test_images() first.")
            return {"error": "No images indexed"}
        if verbose:
            print(f"Images indexed: {count}")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return {"error": str(e)}
    
    # Run tests
    results = []
    for tc in test_cases:
        result = run_single_test(tc, config, verbose)
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model_key,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{100 * passed / len(results):.1f}%" if results else "N/A",
        "results": [r.to_dict() for r in results],
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {passed}/{len(results)} passed ({summary['pass_rate']})")
        print(f"{'='*60}")
    
    if save_results:
        results_file = f"tests/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {results_file}")
    
    return summary


def index_test_images(clear_first: bool = False):
    """Download and index test images to ChromaDB."""
    print("Setting up test images...")
    
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        print("❌ COHERE_API_KEY not set")
        return
    
    # Download images
    downloaded = download_test_images()
    all_paths = []
    for paths in downloaded.values():
        all_paths.extend(paths)
    
    if not all_paths:
        print("❌ No images downloaded")
        return
    
    print(f"\nDownloaded {len(all_paths)} images")
    
    # Index to ChromaDB
    client = get_chromadb_client()
    collection = get_or_create_collection(client)
    
    if clear_first:
        print("Clearing existing index...")
        clear_collection(collection)
    
    print("Embedding images (this may take a minute)...")
    cohere_client = get_cohere_client(cohere_key)
    
    embeddings = embed_images(
        all_paths,
        cohere_client,
        progress_callback=lambda c, t: print(f"  {c}/{t}", end="\r"),
    )
    
    print("\nAdding to ChromaDB...")
    add_images_to_chromadb(collection, all_paths, embeddings)
    
    final_count = get_collection_count(collection)
    print(f"✅ ChromaDB now has {final_count} images indexed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision RAG Test Runner")
    parser.add_argument("--setup", action="store_true", help="Download and index test images")
    parser.add_argument("--clear", action="store_true", help="Clear index before setup")
    parser.add_argument("--run", action="store_true", help="Run test suite")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Filter by difficulty")
    parser.add_argument("--model", default="grok-4.1-fast", help="Model to use")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--limit", type=int, help="Limit number of tests to run")
    parser.add_argument("--id", type=str, help="Run specific test by ID (e.g., fin_001)")
    
    args = parser.parse_args()
    
    if args.setup:
        index_test_images(clear_first=args.clear)
    
    if args.run:
        test_cases = None
        
        # Filter by specific ID
        if args.id:
            test_cases = [tc for tc in TEST_CASES if tc["id"] == args.id]
            if not test_cases:
                print(f"❌ No test found with ID: {args.id}")
                print(f"Available IDs: {[tc['id'] for tc in TEST_CASES[:10]]}...")
                sys.exit(1)
        # Filter by difficulty
        elif args.difficulty:
            test_cases = get_test_cases_by_difficulty(args.difficulty)
        
        # Apply limit
        if args.limit and test_cases:
            test_cases = test_cases[:args.limit]
        elif args.limit:
            test_cases = TEST_CASES[:args.limit]
        
        run_test_suite(
            test_cases=test_cases,
            model_key=args.model,
            verbose=not args.quiet,
        )
    
    if not args.setup and not args.run:
        print("Usage:")
        print("  python run_tests.py --setup        # Download and index test images")
        print("  python run_tests.py --run          # Run all tests")
        print("  python run_tests.py --run --limit 2         # Run first 2 tests only")
        print("  python run_tests.py --run --id fin_001      # Run specific test")
        print("  python run_tests.py --run --difficulty easy # Run easy tests only")
        print("  python run_tests.py --setup --run  # Setup and run")

