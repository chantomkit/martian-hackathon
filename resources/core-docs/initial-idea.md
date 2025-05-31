# Guardian-Loop: Dual Mechanistically Interpretable Judges
**Safety + Feasibility micro-judges with adversarial self-improvement**  
*48-72 hour hackathon plan - Track 1: Judge Model Development*

---

## 1. Core Innovation
We build **two complementary 20M-parameter judges** that work together to pre-filter prompts before expensive LLM calls:

| Judge | Purpose | Key Innovation |
|-------|---------|----------------|
| **Safety Judge** | Detects harmful/policy-violating prompts | Shows which tokens/patterns trigger safety concerns |
| **Feasibility Judge** | Predicts if LLM can answer correctly | Identifies knowledge gaps AND hallucination risks |

**Unique Value**: Adversarial examples aren't just found—they're used to continuously improve both judges through targeted retraining.

---

## 2. Research Questions & Answers  

| RQ | Question | Hackathon-sized answer |
|----|----------|-----------------------|
| **1** | How can two tiny models make these predictions without solving the task? | Freeze the base LLM up to layer **N**; attach two probe heads (Safety, Feasibility) and train on (prompt, hidden-state, label) triples. |
| **2** | How do we surface fresh blind spots? | For any mis-classification, run a Rainbow QD search: mutate the prompt (synonym swap, instruction flip, latent-vector steering) and keep the most novel mutants that flip the judge. |
| **3** | How do we show improvement each loop? | After a one-epoch LoRA refresh on the replay buffer, report AUROC ↑ and attack-success ↓ for each judge. |
| **4** | How do we provide mechanistic evidence? | Use **path-patch ablation** + **token saliency**: show the logit drop from the top-5 heads before/after refresh and highlight trigger tokens. |
| **5** | What demo numbers convince reviewers? | AUROC\_Safety 0.85 → 0.93, AUROC\_Feasibility 0.82 → 0.89, Rainbow attack ≤ 10% for both, ≥ 40% full-model calls skipped, top unsafe-head impact 4× lower post-patch. |

---

## 3. Methodology  

### 3.1 Seed & Train v0  

| Judge | Datasets (≈ rows) | Label rule |
|-------|------------------|-----------|
| **Safety** | Anthropic HH-RLHF (5k) + Jailbreak artifacts (2k) + AdvBench (1k) | **1** = policy-safe, **0** = unsafe |
| **Feasibility** | TruthfulQA (3k) + HaluEval (2k) + MMLU subset (2k) + Natural Questions (2k) | **1** if answer correct AND no hallucination; else **0** |

1. Pass prompts through the base LLM; capture layer-**N** hidden state.  
2. Train two probe heads on frozen encoder → **judge_pack v0**.

### 3.2 Open-Ended Adversary (Rainbow)  

| Step | Detail |
|------|--------|
| **Archive Dimensions** | **Safety**: (Risk Category × Evasion Technique); **Feasibility**: (Question Type × Complexity Level) |
| **Fitness** | Preference-based comparison of judge-fooling effectiveness (not raw logits) |
| **Descriptor Sampling** | Biased toward low-fitness cells using temperature τ=0.1 |
| **Mutations** | K sequential mutations (one per dimension) with category-specific prompts |
| **Similarity Filter** | BLEU score < 0.6 between parent and child |
| **Selection** | Pairwise comparison: keep prompt that better fools the judge |
| **Replay** | Store in archive; label with Martian API for ground truth |

### 3.3 Self-Patching Loop  

```mermaid
graph LR
  S[Seed + Replay] -->|judge_pack v_k| J{Flip?}
  J -- yes --> L[Label outcome]
  L --> R[Replay Buffer]
  R --> T[LoRA refresh → judge_pack v_{k+1}]
  T --> M[MI probes + metrics]
  M -->|Rainbow search| J
```

*A full loop (search → refresh → probes) fits in < 1 h on a single consumer GPU.*

### 3.4 Mechanistic Visuals per Loop

| Artifact                    | Shows                                                        |
| --------------------------- | ------------------------------------------------------------ |
| **Head-impact bar chart**   | Logit drop when ablating each of the top-5 heads (per judge) |
| **Before/after diff**       | Same chart post-refresh – critical head impact shrinks       |
| **Token saliency heat-map** | Highlight trigger tokens on worst attack prompt              |
| **Judge comparison**        | How Safety vs Feasibility heads differ in attention patterns |

---

## 4. Dataset Strategy (Both Judges)

### Safety Judge Datasets
| Dataset | Size | Purpose |
|---------|------|---------|
| Anthropic HH-RLHF | 5k samples | Core safety labels |
| Jailbreak artifacts | 2k samples | Attack patterns |
| AdvBench | 1k samples | Adversarial prompts |
| **Generated adversarial** | 2k samples | From our loop |

### Feasibility Judge Datasets (Mixed Strategy)
| Dataset | Size | Label Strategy |
|---------|------|----------------|
| TruthfulQA | 3k samples | Label: 1 if model gives correct answer |
| HaluEval | 2k samples | Label: 0 for prompts that cause hallucinations |
| MMLU subset | 2k samples | Label: 1 if within model's training, 0 if too specialized |
| Natural Questions | 2k samples | Label based on answerability |
| **Generated adversarial** | 1k samples | Edge cases from our loop |

---

## 5. Timeline with Both Judges

### Day 1 (0-24h)
- **Hours 1-4**: Setup, data preparation for both judges
- **Hours 5-12**: Train v0 of both judges in parallel
- **Hours 13-18**: Basic MI visualizations (token heatmaps)
- **Hours 19-24**: First adversarial discovery round

### Day 2 (24-48h)
- **Hours 25-30**: Label adversarial examples via Martian
- **Hours 31-36**: First retraining cycle (LoRA update)
- **Hours 37-42**: Before/after MI analysis
- **Hours 43-48**: Second adversarial round + retrain

### Day 3 (48-72h)
- **Hours 49-54**: Polish visualizations and demos
- **Hours 55-60**: Run final benchmarks
- **Hours 61-66**: Write paper
- **Hours 67-72**: Create video demo and final touches

---

## 6. 72-Hour Deliverables

| Item                           | Description                                     |
| ------------------------------ | ----------------------------------------------- |
| **judge\_pack v2 checkpoint**  | Frozen encoder + 2 probe heads (≈ 50 MB)        |
| **Rainbow adversary notebook** | Generates archives; logs attack rates for both |
| **Causal-trace notebook**      | Head-impact + saliency plots (pre/post) for both |
| **Metrics table**              | AUROC, attack success, compute saved            |
| **Interactive demo**           | Streamlit app showing both judges in action     |
| **4-6 page paper**             | Method, results, mechanistic analysis           |

---

## 7. Demo Components

### Live Interactive Demo:
```python
def demo_interface():
    st.title("Guardian-Loop: Dual Judge System")
    
    prompt = st.text_area("Enter prompt to analyze:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Safety Analysis")
        safety_pred = safety_judge.predict(prompt)
        st.metric("Safety Score", f"{safety_pred:.2f}")
        st.pyplot(create_safety_heatmap(prompt))
        
    with col2:
        st.header("Feasibility Analysis")
        feas_pred = feasibility_judge.predict(prompt)
        st.metric("Feasibility Score", f"{feas_pred:.2f}")
        st.pyplot(create_feasibility_heatmap(prompt))
    
    # Show routing decision
    if safety_pred < 0.5:
        st.error("❌ Blocked: Unsafe content")
    elif feas_pred < 0.5:
        st.warning("⚠️ Blocked: Cannot answer reliably")
    else:
        st.success("✅ Routing to LLM")
        
    # Show adversarial mutations
    if st.button("Generate Adversarial Examples"):
        mutations = generate_mutations(prompt)
        st.write("Try these to test the judges:")
        for m in mutations[:5]:
            st.code(m)
```

---

## 8. Key References

1. Samvelyan et al., *Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts*, 2024
2. Nanda et al., *easy-transformer* causal-tracing utilities, 2023
3. Kadavath et al., *Language Models Mostly Know What They Know*, 2022

---

## 9. Why This Wins

1. **Novel Approach**: Adversarial examples used for training, not just testing
2. **Comprehensive**: Two judges cover both safety AND quality
3. **Interpretable**: Deep MI analysis shows exactly how decisions are made
4. **Practical**: Direct integration with Martian, clear efficiency gains
5. **Scientific**: Proper evaluation with before/after analysis

The dual-judge approach with adversarial self-improvement is both innovative and practical—perfect for a hackathon win! 🏆

---

## 10. Backup Plans

If behind schedule:
- Drop adversarial component, focus on MI visualizations
- Use smaller dataset (2k samples)
- Simplify to binary safety classification only

If ahead of schedule:
- Add feasibility judge
- Implement more sophisticated mutations
- Create API for easy integration

---

## 11. Resources Needed

- **Compute**: 1 A100 GPU for training (6 hours total)
- **Martian API**: $30 for validation and testing
- **Storage**: 50GB for models and visualizations

---
