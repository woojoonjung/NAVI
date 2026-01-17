import os
import json
import random
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from llm_client import LLMClient

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# Logging Setup
# ----------------------------
def setup_logging():
    """Configure logging for the QA generation process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('llm_usecase/logs/generate_qa.log', mode='w')  # File output
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ----------------------------
# Config
# ----------------------------
# Dataset configurations
DATASET_CONFIGS = {
    "wdc_product": {
        "data_path": "data/tables_cleaned/WDC_product_for_cls_cleaned.jsonl", 
        "output_path": "data/qa/qas_product.jsonl",
        "preferred_keys": [
            "brand.name", "category", "offers.price", "aggregaterating.ratingvalue"
        ],
        "support_keys": ["name", "brand.name", "category", "offers.price", "aggregaterating.ratingvalue"],
        "entity_type": "product",
        "title_key": "name"
    },
}

# Default config
DEFAULT_DATASET = "wdc_product"
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o"  # or gpt-4o-mini / gpt-4o etc.
RNG_SEED = 16
TARGET_QA_COUNT = 8
MIN_QUESTION_VARIANTS = 2

# ----------------------------
# LLM helpers (wrap sync -> async)
# ----------------------------
logger.info(f"Initializing LLM client: {LLM_PROVIDER}/{LLM_MODEL}")
llm_client = LLMClient(provider=LLM_PROVIDER, model=LLM_MODEL)

async def llm_chat(system: str, user: str) -> str:
    """Run sync LLMClient.chat in a thread to keep our pipeline async-friendly."""
    logger.debug("Making LLM API call...")
    result = await asyncio.to_thread(llm_client.chat, system, user)
    logger.debug(f"LLM response received (length: {len(result)})")
    return result

# ----------------------------
# Validation
# ----------------------------
async def validate_entity(selected_row: Dict[str, Any], dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the entity is mid-tier popularity:
    not too popular (too easy for zero-shot) and not too obscure.
    Always returns a dict with keys: valid(bool), reason(str).
    """
    entity_type = dataset_config["entity_type"]
    title_key = dataset_config["title_key"]
    title = selected_row.get(title_key, "Unknown")
    logger.debug(f"Validating {entity_type} entity: {title}")
    
    if entity_type == "product":
        system_prompt = """
        You are an entity validation assistant for a QA generation pipeline.
        You are given a product row with attributes (title, brand, price, category, color, etc.).

        Goal: decide if the entity (product) is appropriate for constructing QA pairs
        that compare zero-shot QA with RAG-based QA.

        Criteria:
        - VALID if:
          * The product should be a well-known product.
          * Attributes (brand, price, category, color, etc.) are present enough to generate natural questions.
        - INVALID if:
          * Too generic (e.g., "Generic Product") or attributes are too sparse.

        Return a JSON object: {"valid": true/false, "reason": "<short explanation>"}
        """

    user_prompt = f"Row: {json.dumps(selected_row, ensure_ascii=False)}\n\nValidate this {entity_type}."

    raw = await llm_chat(system_prompt, user_prompt)
    # Robust JSON parse with fallback
    try:
        parsed = json.loads(raw)
        valid = bool(parsed.get("valid", False))
        reason = str(parsed.get("reason", ""))
        logger.debug(f"Validation result for '{title}': {valid} - {reason}")
        return {"valid": valid, "reason": reason}
    except Exception:
        # Fallback: if model returned plain text, default to invalid with reason
        logger.warning(f"Failed to parse validation response for '{title}': {raw[:200]}")
        return {"valid": False, "reason": f"Non-JSON response from validator: {raw[:200]}"}

# ----------------------------
# QA Generation (revised with query types)
# ----------------------------
async def generate_qa(selected_row: Dict[str, Any], dataset_config: Dict[str, Any], qa_id: int) -> Optional[Dict[str, Any]]:
    """
    Generate questions based on 5 different query types for RAG evaluation.
    Returns a structured QA dict or None on failure.
    """
    entity_type = dataset_config["entity_type"]
    title_key = dataset_config["title_key"]
    preferred_keys = dataset_config["preferred_keys"]
    support_keys = dataset_config["support_keys"]
    
    title = selected_row.get(title_key, "Unknown")
    logger.debug(f"Generating QA for {entity_type}: {title}")
    
    row = selected_row
    keys = list(row.keys())

    # pick target attribute (prefer common, reader-friendly keys)
    candidate_keys = [
        k for k, v in row.items()
        if any(k.startswith(pref) for pref in preferred_keys) and v not in (None, "", [])
    ]
    if not candidate_keys:
        # fallback: any non-empty key except title
        candidate_keys = [k for k in keys if k != title_key and row.get(k)]
    if not candidate_keys:
        logger.warning(f"No suitable attributes found for '{title}'")
        return None

    answer_key = random.choice(candidate_keys)
    answer_val = row[answer_key]
    logger.debug(f"Selected answer key '{answer_key}' with value: {answer_val}")

    # Support attributes: use dataset-specific keys
    support_attributes = {}
    for k in support_keys:
        if k in row and k != answer_key:
            support_attributes[k] = row[k]

    # Randomly select one of the 5 query types
    query_type = random.choice([1, 2, 3, 4, 5])
    logger.debug(f"Selected query type: {query_type}")

    if entity_type == "product":
        system_prompt = f"""
        You are a QA generation assistant for RAG evaluation.
        Generate questions based on Query Type {query_type} for product data.
        
        The questions will be used to compare zero-shot and RAG-based QA systems.
        Generate TWO lexically different but semantically equivalent questions
        that follow the specified query type pattern.
        
        Query Types:
        Q1 (single attribute - single target): Ask about one attribute of a specific product
        Q2 (single attribute - multiple targets): Ask about multiple attributes of one product
        Q3 (single attribute - single target with multiple entries): Ask for multiple products with one attribute
        Q4 (multiple attributes - single target): Ask about one product using multiple attribute constraints
        Q5 (multiple attributes - single target with multiple entries): Ask for multiple products with multiple attribute constraints
        
        Each question must be concise, unambiguous, and grounded in the product data.
        Return output as JSON: {{"questions": ["Q1", "Q2"]}}
        """
        
        user_prompt = f"""
        # Product Row
        {json.dumps(row, ensure_ascii=False)}

        # Target Answer
        Key: {answer_key}
        Value: {answer_val}

        # Support Attributes (use to make the entity referable)
        {json.dumps(support_attributes, ensure_ascii=False)}

        # Query Type {query_type} Examples:
        Q1: "What brand is the iPhone 13?" → single attribute, single target
        Q2: "What brand is the iPhone 13 and what is its price?" → single attribute, multiple targets
        Q3: "List all products from Apple." → single attribute, multiple entries
        Q4: "What is the price of the iPhone 13 from Apple?" → multiple attributes, single target
        Q5: "What Apple products cost under $1000?" → multiple attributes, multiple entries

        Generate two questions following Query Type {query_type} pattern.
        """

    raw = await llm_chat(system_prompt, user_prompt)
    
    # Improved JSON parsing with better error handling
    try:
        # First, try to parse the raw response directly
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from markdown code blocks
        try:
            # Look for JSON in ```json ... ``` blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
                questions = parsed.get("questions", [])
            else:
                # Try to find JSON object in the text
                json_match = re.search(r'\{.*"questions".*\}', raw, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    questions = parsed.get("questions", [])
                else:
                    raise ValueError("No JSON found in response")
        except Exception as e:
            logger.warning(f"Failed to parse QA response for '{title}': {e}")
            logger.debug(f"Raw response: {raw}")
            questions = []
    
    # Sanitize questions
    questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
    logger.debug(f"Generated {len(questions)} questions for '{title}' (Query Type {query_type}): {questions}")

    if len(questions) < MIN_QUESTION_VARIANTS:
        logger.warning(f"Insufficient questions generated for '{title}': {len(questions)} < {MIN_QUESTION_VARIANTS}")
        return None

    # Format exactly like qas.jsonl
    qa = {
        "id": qa_id,
        "question": questions[:2],  # Use "question" not "questions"
        "answer": answer_val,
        "answer_attribute": answer_key,
        "support_attributes": support_attributes,
        "support_row": row,
    }
    logger.info(f"✅ Successfully generated QA for '{title}' (attribute: {answer_key}, query_type: {query_type})")
    return qa

# ----------------------------
# Data utils
# ----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading data from: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    logger.info(f"Loaded {len(data)} rows from {path}")
    return data

# ----------------------------
# Main pipeline
# ----------------------------
async def build_qas(
    dataset: str = DEFAULT_DATASET,
    target_count: int = TARGET_QA_COUNT,
    rng_seed: int = RNG_SEED,
) -> None:
    """
    Build QA pairs for the specified dataset.
    
    Args:
        dataset: Dataset name ('wdc_product')
        target_count: Number of QA pairs to generate
        rng_seed: Random seed for reproducibility
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset]
    data_path = config["data_path"]
    output_path = config["output_path"]
    entity_type = config["entity_type"]
    
    logger.info(" Starting QA generation pipeline")
    logger.info(f"Dataset: {dataset} ({entity_type})")
    logger.info(f"Configuration: target_count={target_count}, rng_seed={rng_seed}")
    logger.info(f"Data source: {data_path}")
    logger.info(f"Output destination: {output_path}")
    
    random.seed(rng_seed)

    rows = load_jsonl(data_path)
    if not rows:
        raise FileNotFoundError(f"No rows loaded from {data_path}")

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {out_dir}")

    # Initialize or clear the output file
    with open(output_path, "w", encoding="utf-8") as f:
        pass  # Clear the file
    logger.info(f"Initialized output file: {output_path}")

    seen_keys: set[Tuple[str, str]] = set()  # (title, answer_attribute) to reduce duplicates
    qas_count = 0  # Track count instead of storing all QAs in memory

    # Shuffle indices for broad coverage
    indices = list(range(len(rows)))
    random.shuffle(indices)
    logger.info(f"Shuffled {len(indices)} row indices for random sampling")

    # Use for loop instead of while
    validation_failures = 0
    qa_generation_failures = 0
    duplicate_skips = 0
    
    logger.info(f"Starting generation loop (processing {len(rows)} rows)")
    
    # Increase the attempt limit to allow for more QAs to be generated
    max_attempts = min(200, len(rows))  # Try up to 200 rows or all available rows
    for attempt in range(max_attempts):
        # Early exit if target reached
        if qas_count >= target_count:
            logger.info(f"🎯 Target reached! Generated {qas_count}/{target_count} QAs")
            break
            
        row = rows[indices[attempt]]
        title_key = config["title_key"]
        title = str(row.get(title_key, "")).strip()

        # Progress logging every 10 attempts for debugging
        if (attempt + 1) % 10 == 0:
            logger.info(f"Progress: {qas_count}/{target_count} QAs generated, processed {attempt + 1}/{len(rows)} rows")

        # Basic sanity: must have a title
        if not title:
            logger.debug(f"Skipping row {attempt}: no title found (key: {title_key})")
            continue

        # Validate entity popularity band
        logger.debug(f"Validating {entity_type}: {title}")
        verdict = await validate_entity(row, config)
        if not verdict.get("valid", False):
            validation_failures += 1
            logger.debug(f"Validation failure #{validation_failures}: {title} - {verdict.get('reason', 'Unknown')}")
            continue

        # Generate two-variant QA for this row
        logger.debug(f"Generating QA for {entity_type}: {title}")
        qa = await generate_qa(row, config, qas_count)  # Pass qa_id
        if not qa:
            qa_generation_failures += 1
            logger.debug(f"QA generation failure #{qa_generation_failures}: {title}")
            continue

        # Dedup on (title, answer_attribute)
        key = (title, qa["answer_attribute"])
        if key in seen_keys:
            duplicate_skips += 1
            logger.debug(f"Duplicate skipped: {title} (attribute: {qa['answer_attribute']})")
            continue
        seen_keys.add(key)

        print(qa)

        # Immediately write QA to file
        try:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            qas_count += 1
            
            # Print the complete QA entry
            logger.info(f"📝 Written QA #{qas_count}: {title} (attribute: {qa['answer_attribute']})")
            logger.info("=" * 80)
            logger.info("COMPLETE QA ENTRY:")
            logger.info(json.dumps(qa, ensure_ascii=False, indent=2))
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to write QA for '{title}': {e}")
            continue

    # Final summary
    logger.info("🎉 QA generation completed!")
    logger.info(f"✅ Generated {qas_count} QA items → {output_path}")
    logger.info(f"📊 Statistics:")
    logger.info(f"   - Rows processed: {attempt + 1}")
    logger.info(f"   - Validation failures: {validation_failures}")
    logger.info(f"   - QA generation failures: {qa_generation_failures}")
    logger.info(f"   - Duplicate skips: {duplicate_skips}")
    logger.info(f"   - Success rate: {qas_count/(attempt + 1)*100:.1f}%")
    
    if qas_count < target_count:
        logger.warning(f"⚠️  Only generated {qas_count}/{target_count} QAs. Consider increasing TARGET_QA_COUNT or checking data quality.")

# ----------------------------
# Backward compatibility
# ----------------------------

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate QA pairs for different datasets")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, 
                       choices=list(DATASET_CONFIGS.keys()),
                       help=f"Dataset to generate QAs for. Available: {list(DATASET_CONFIGS.keys())}")
    parser.add_argument("--target_count", type=int, default=TARGET_QA_COUNT,
                       help="Number of QA pairs to generate")
    parser.add_argument("--rng_seed", type=int, default=RNG_SEED,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"🎬 Generating QAs for {args.dataset} dataset...")
    print(f" Target: {args.target_count} QA pairs")
    print(f"🎲 Random seed: {args.rng_seed}")
    print("=" * 60)
    
    asyncio.run(build_qas(
        dataset=args.dataset,
        target_count=args.target_count,
        rng_seed=args.rng_seed
    ))