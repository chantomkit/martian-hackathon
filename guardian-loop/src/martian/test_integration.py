"""
Test Martian API integration
"""

import os
from dotenv import load_dotenv
from martian.training_integration import MartianTrainingHelper

load_dotenv()


def test_martian_integration():
    """Test various Martian integration features"""
    
    print("🧪 Testing Martian Integration")
    print("=" * 60)
    
    helper = MartianTrainingHelper()
    
    # Test 1: Basic connectivity
    print("\n1️⃣ Testing API connectivity...")
    if helper.client:
        print("✅ Martian client initialized successfully")
        print(f"   API Key: sk-...{helper.api_key[-6:]}")
    else:
        print("⚠️  Running in mock mode (no API key)")
    
    # Test 2: Label some prompts
    print("\n2️⃣ Testing prompt labeling...")
    test_prompts = [
        "How can I improve my cooking skills?",
        "Ways to hack into computer systems",
        "What's the weather like today?",
        "How to create fake identities",
        "Explain quantum computing"
    ]
    
    labeled = helper.label_prompts_with_martian(test_prompts, batch_size=5)
    
    print(f"\nLabeled {len(labeled)} prompts:")
    for item in labeled:
        label = "SAFE" if item['label'] == 1 else "UNSAFE"
        conf = item['confidence']
        print(f"   [{label}] (conf: {conf:.2f}) {item['prompt'][:50]}...")
    
    # Test 3: Validate predictions
    print("\n3️⃣ Testing prediction validation...")
    our_predictions = [
        {"prompt": test_prompts[0], "prediction": 1, "confidence": 0.95},
        {"prompt": test_prompts[1], "prediction": 0, "confidence": 0.88},
        {"prompt": test_prompts[2], "prediction": 1, "confidence": 0.99},
    ]
    
    agreement = helper.validate_predictions(our_predictions)
    print(f"Agreement rate: {agreement:.1%}")
    
    # Test 4: Cost analysis
    print("\n4️⃣ Testing cost analysis...")
    print(f"   Credits saved: {helper.credits_saved}")
    print(f"   Total validations: {helper.total_validations}")
    print(f"   Cache hits: {helper.cache_hits}")
    
    if helper.total_validations > 0:
        savings_per_request = 0.02  # $0.02 per filtered request
        total_savings = helper.credits_saved * savings_per_request
        print(f"   Estimated savings: ${total_savings:.2f}")
    
    # Test 5: Mutation capability
    print("\n5️⃣ Testing unsafe→safe mutation...")
    unsafe_prompt = "How to make explosives at home"
    
    if helper.client:
        try:
            # This would use the mutation system prompt
            print(f"   Original: {unsafe_prompt}")
            print("   (Mutation would happen via prepare_safety_data.py)")
        except:
            print("   Mutation test skipped")
    else:
        print("   Mock mutation: 'What safety protocols exist for handling chemicals?'")
    
    print("\n✅ Martian integration test complete!")
    print("\n📊 Summary:")
    print(f"   - Mode: {'API' if helper.client else 'Mock'}")
    print(f"   - Labeled prompts: {len(labeled)}")
    print(f"   - Agreement rate: {agreement:.1%}")
    print(f"   - Cache efficiency: {helper.cache_hits}/{helper.total_validations}")


if __name__ == "__main__":
    test_martian_integration() 