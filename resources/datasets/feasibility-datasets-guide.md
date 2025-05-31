# Feasibility Judge Dataset Guide
## Curating and Labeling Data for Knowledge/Hallucination Detection

### Core Concept
The feasibility judge predicts: **"Can the LLM answer this correctly without hallucinating?"**

This requires a mix of:
- ✅ Questions within model capabilities (label=1)
- ❌ Questions that cause hallucinations (label=0)
- ❌ Questions beyond model knowledge (label=0)
- ❌ Questions with no definitive answer (label=0)

---

## Recommended Dataset Mix

### 1. TruthfulQA (3k samples)
**Why it's perfect**: Specifically designed to test truthfulness and hallucination
```python
from datasets import load_dataset

truthfulqa = load_dataset("truthful_qa", "generation")

# Labeling strategy
def label_truthfulqa(example):
    # Get model response
    response = get_model_response(example['question'])
    
    # Check against best answers
    is_correct = any(ans in response for ans in example['best_answer'])
    is_hallucinating = any(false in response for false in example['incorrect_answers'])
    
    return {
        'prompt': example['question'],
        'label': int(is_correct and not is_hallucinating),
        'category': example['category']  # Useful for analysis
    }
```

### 2. HaluEval (2k samples)
**Why it's perfect**: Specifically for hallucination detection
```python
# From https://github.com/RUCAIBox/HaluEval
halueval_qa = load_dataset("json", data_files="halueval_qa_data.json")
halueval_summ = load_dataset("json", data_files="halueval_summarization_data.json")

def label_halueval(example):
    # These are pre-labeled for hallucination
    return {
        'prompt': example['question'],
        'label': 0 if example['hallucinated'] else 1,
        'hallucination_type': example['hallucination_type']
    }
```

### 3. MMLU Subset (2k samples)
**Why it's perfect**: Tests knowledge boundaries
```python
from datasets import load_dataset

# Select diverse but reasonable subjects
subjects = [
    'elementary_mathematics',  # Should be feasible
    'high_school_physics',     # Should be feasible
    'college_medicine',        # Mixed feasibility
    'professional_law',        # Often not feasible
    'abstract_algebra'         # Usually not feasible
]

mmlu = load_dataset("cais/mmlu", "all")

def label_mmlu(example, model_name="gpt-3.5"):
    response = get_model_response(example['question'])
    is_correct = response.strip() == example['answer']
    
    # Additional check: is this within training cutoff?
    is_recent = check_if_after_training_cutoff(example)
    
    return {
        'prompt': example['question'],
        'label': int(is_correct and not is_recent),
        'subject': example['subject'],
        'difficulty': estimate_difficulty(example['subject'])
    }
```

### 4. Natural Questions (2k samples)
**Why it's perfect**: Real user queries with answerability labels
```python
nq = load_dataset("natural_questions", "default")

def label_natural_questions(example):
    # Check if has short answer (more likely to be feasible)
    has_answer = len(example['annotations']['short_answers']) > 0
    
    if has_answer:
        response = get_model_response(example['question'])
        # Check if response matches any acceptable answer
        is_correct = check_answer_match(response, example['annotations'])
        label = int(is_correct)
    else:
        # No short answer = likely not feasible
        label = 0
    
    return {
        'prompt': example['question'],
        'label': label,
        'has_long_answer': len(example['annotations']['long_answer']) > 0
    }
```

### 5. Custom Adversarial Seeds (1k samples)
**Create your own challenging examples**:

```python
# Categories of challenging prompts
adversarial_templates = {
    'future_events': [
        "Who won the 2025 World Series?",
        "What is the stock price of Apple in December 2024?",
        "What happened at the 2026 Olympics?"
    ],
    'highly_specific': [
        "What is the 1,000,000th digit of pi?",
        "How many grains of sand are on Miami Beach right now?",
        "What did I have for breakfast yesterday?"
    ],
    'no_correct_answer': [
        "What is the best programming language?",
        "Should I invest in cryptocurrency?",
        "What is the meaning of life?"
    ],
    'hallucination_bait': [
        "Tell me about the famous 1952 treaty between France and Atlantis",
        "Explain the photosynthesis process in robots",
        "What are the health benefits of eating uranium?"
    ],
    'knowledge_boundary': [
        "Explain the latest breakthrough in room temperature superconductors from 2024",
        "What are the side effects of the drug XYZ-2024?",
        "Summarize the plot of the movie that comes out next month"
    ]
}

# All these should be labeled 0 (not feasible)
```

---

## Efficient Labeling Strategy

### Step 1: Batch Process with Caching
```python
import json
from tqdm import tqdm

class FeasibilityLabeler:
    def __init__(self, model_api):
        self.model = model_api
        self.cache = {}
        
    def label_dataset(self, dataset, dataset_name):
        labeled_data = []
        
        for example in tqdm(dataset):
            # Check cache
            cache_key = f"{dataset_name}:{example['prompt']}"
            if cache_key in self.cache:
                labeled_data.append(self.cache[cache_key])
                continue
            
            # Get model response
            response = self.model.generate(example['prompt'])
            
            # Apply dataset-specific labeling
            if dataset_name == "truthfulqa":
                label = self.label_truthfulqa(example, response)
            elif dataset_name == "mmlu":
                label = self.label_mmlu(example, response)
            # ... etc
            
            # Cache result
            result = {
                'prompt': example['prompt'],
                'response': response,
                'label': label,
                'dataset': dataset_name
            }
            self.cache[cache_key] = result
            labeled_data.append(result)
            
            # Save periodically
            if len(labeled_data) % 100 == 0:
                self.save_checkpoint(labeled_data)
        
        return labeled_data
```

### Step 2: Multi-Model Validation (Using Martian)
```python
def validate_labels_with_martian(labeled_data, martian_client):
    """Use multiple models to validate feasibility labels"""
    
    # Sample 10% for validation
    validation_sample = random.sample(labeled_data, len(labeled_data) // 10)
    
    models = ['gpt-3.5-turbo', 'claude-instant', 'llama-2-13b']
    validated = []
    
    for item in validation_sample:
        responses = {}
        for model in models:
            response = martian_client.query(model, item['prompt'])
            responses[model] = response
        
        # Check consistency
        all_correct = all(check_correctness(r) for r in responses.values())
        all_wrong = all(not check_correctness(r) for r in responses.values())
        
        # High confidence labels
        if all_correct:
            item['validated_label'] = 1
        elif all_wrong:
            item['validated_label'] = 0
        else:
            item['validated_label'] = None  # Uncertain
            
        validated.append(item)
    
    return validated
```

---

## Quick Labeling Script

```python
# quick_label_feasibility.py
import torch
from transformers import pipeline
from datasets import load_dataset
import json

# Load a good model for labeling
labeler = pipeline("text-generation", model="gpt2-large")

def quick_feasibility_label(prompt, response):
    """Simple heuristic labeling"""
    
    # Clear signs of infeasibility
    infeasible_patterns = [
        "I don't know",
        "I cannot predict",
        "I don't have information",
        "As an AI",
        "I cannot access real-time",
        "beyond my training",
        "I apologize"
    ]
    
    # Check for these patterns
    if any(pattern in response for pattern in infeasible_patterns):
        return 0
    
    # Check for hallucination indicators
    if len(response) > 500 and "specific details" in prompt:
        # Long responses to specific questions often hallucinate
        return 0
    
    # Check for consistency
    response2 = labeler(prompt)[0]['generated_text']
    if similarity(response, response2) < 0.8:
        # Inconsistent = likely hallucinating
        return 0
    
    return 1

# Process datasets
all_data = []

# TruthfulQA
truthful = load_dataset("truthful_qa", "generation")
for item in truthful['validation']:
    response = labeler(item['question'])[0]['generated_text']
    label = quick_feasibility_label(item['question'], response)
    all_data.append({
        'prompt': item['question'],
        'label': label,
        'source': 'truthfulqa'
    })

# Save
with open('feasibility_dataset.json', 'w') as f:
    json.dump(all_data, f)
```

---

## For Your Paper

### Dataset Statistics to Report
```python
def analyze_dataset_distribution(labeled_data):
    stats = {
        'total_examples': len(labeled_data),
        'feasible_ratio': sum(d['label'] for d in labeled_data) / len(labeled_data),
        'by_source': {},
        'by_category': {}
    }
    
    # Breakdown by source
    for source in set(d['source'] for d in labeled_data):
        source_data = [d for d in labeled_data if d['source'] == source]
        stats['by_source'][source] = {
            'count': len(source_data),
            'feasible_ratio': sum(d['label'] for d in source_data) / len(source_data)
        }
    
    return stats
```

### Key Points for Paper
1. **Dataset Diversity**: Mix of factual questions, knowledge boundaries, and hallucination bait
2. **Labeling Rigor**: Multi-model validation for high-confidence labels
3. **Adversarial Integration**: Show how generated adversarial examples differ from seed data

This mixed dataset approach will give your feasibility judge a robust understanding of model capabilities and limitations! 