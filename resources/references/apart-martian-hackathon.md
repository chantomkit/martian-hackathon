## Apart x Martian Mechanistic Router Interpretability Hackathon

*Join the effort to make AI orchestration interpretable from the ground up—where judge models reveal their reasoning process and routing decisions become windows into AI decision-making!*



### Overview

🧐 Use mechanistic interpretability to evaluate expert models and improve trustworthiness and transparency.

Come along to hack together new methods to create safer and more secure routing of requests to specialized expert models!

Monolithic AI models like GPT-4o and Claude face basic limitations in transparency and widespread access, and their use requires us to rely on highly capable general models instead of expert systems that we can trust and verify to a much higher degree.

We're joined by **Martian** who have developed the **Expert Orchestration AI Architecture**, an exciting technology to route requests according to key criteria for alignment, verifiability, and reliability. Here, "judge" models evaluate the capabilities of expert models and "router" systems direct queries to the most trustworthy experts based on these evaluations.

In this hackathon, we'll develop new mechanistic interpretability techniques that rely on novel model understanding to create safer, more transparent deployment of models.
You will also receive **$50 in Martian API credits** to execute on your exciting idea - Sign up above to stay updated!

---

### Why This Hackathon Matters:

The current trajectory of AI development, focused on increasingly powerful monolithic models, faces fundamental limitations:

* **Winner-takes-all dynamics:** The high costs of training frontier models leads to market concentration and economic power in a few corporations.
* **Misaligned safety incentives:** Companies racing to release increasingly powerful models may underreport risks and rush products to market.
* **High barriers to entry:** Specialized models struggle to gain market traction against generalist models, even when they excel in specific domains.
* **Limited user control:** Users have minimal visibility into how models "think" and limited ability to control characteristics like factuality, bias, or ethical reasoning.
* **Inefficient resource use:** Using powerful frontier models for all tasks wastes resources and often underperforms specialized alternatives.

The Expert Orchestration Architecture addresses these issues by creating a more transparent, efficient, and democratic AI ecosystem where specialized models can thrive based on their unique strengths, and users gain unprecedented control over AI capabilities.

---

### Expected Outcomes

Participants will create components that advance the Expert Orchestration vision:

* Prototype judge models for evaluating specific AI capabilities
* Intelligent routing algorithms for directing queries to appropriate models
* Frameworks for decomposing complex tasks across multiple specialized models
* Integration APIs that allow seamless discovery and utilization of specialized models
* Evaluation metrics and benchmarks for comparing different routing and judge strategies

The most promising projects will have opportunities for continued development and potential integration into production systems.

---

### Challenge Tracks

#### Track 1: Judge Model Development
Build specialized evaluation models that can assess different AI models' capabilities across dimensions that matter to users (manipulation skills and tendencies, deception and hidden communication, misaligned goals, factuality, domain expertise, ethics, creativity, objectivity, etc.). Judges should provide independent, objective evaluations that create transparency around model strengths and weaknesses.

#### Track 2: Intelligent Router Systems
Develop router systems that can intelligently direct user queries to the most appropriate specialized models based on user preferences using judge evaluations. Focus areas include routers that use multiple judges (e.g. factuality, ethics and lack of deception), query classification, efficiency optimization, and handling uncertainty.

#### Track 3: Task Decomposition Frameworks
Create systems that can break down complex user requests into a series of more manageable steps, to be executed by different specialized models. This includes planning phases, execution phases, and coordination mechanisms. Investigate whether decomposition avoids or reduces some of the traps reported for monolithic reasoning models (e.g. reward hacking).

#### Track 4: Specialized Model Integration
Build frameworks that enable easy integration of new specialized models into the Expert Orchestration Architecture, including methods for model discovery, capability profiling, and dynamic performance evaluation.

---

### Open Research Questions

#### Judges

* **Model Characteristic Analysis:** Create dataset(s) that test an interesting model characteristic pertinent to safety (e.g., ethics, hallucinations, gender bias). Build a judge using this data and evaluate multiple models.
* **Judge Evaluation Metrics:** Develop methods to measure judge accuracy, completeness, and reliability for specific characteristics. Explore how this impacts AI Safety.
* **Mechanistic Interpretability for Judges:** Apply MI techniques to model internals to create better or more interpretable judges e.g. judges that can evaluate outputs based on how they were generated.
* **Measuring model similarity:** HuggingFace hosts thousands of LLMs. How do we measure whether two models have similar capabilities? How do we choose a subset of these models with a sufficiently diverse set of capabilities, that after training the resulting router will be performant? How does the router performance vary with the size of the subset?

#### Routers
Given a set of models with known capabilities measured by known judge scores:

* **Risk-Sensitive Routing:** Build efficient routing algorithms balancing the judge scores, computes costs, and system reliability for the best user experience.
* **Multi-Objective Routing:** Create routers that use scores from multiple judges (e.g., answer correctness, ethics and legality) according to user preferences for the best user experience. What are the tradeoffs?
* **Routing algorithms:** For expensive models, the judge provides a “pre-hoc” way to estimate prediction success (without querying the model). For cheap models, we can ask the model to evaluate the answer, and evaluate its confidence (“post-hoc”) in its predictions. Find interesting ways to mix pre- and post-hoc routing to get the best of both worlds.
* **Multi-level routing:** Investigate using a tree of choices rather than one-off routing. What are the pros and cons?
* **Reducing router training costs:** Given a model and a task, how can we cheaply detect a model is not a good fit for a task - avoiding further training time optimizing how bad a fit it is.
* **Task Decomposition:** Model breaking a complex user task into multiple subtasks that can be routed to the most capable models before recombining the results. What are the AI Safety, cyber-security and/or cost implications of this approach?
* **Universal router:** For a set of tasks, create a single router across a set of LLMs that provides higher-quality answers than any single LLM does.

#### Inferring Judges & Routers

* **Reverse Engineering:** Given a black-box LLM or router, infer its embedded judge (reward signal) for specific characteristics.
* **Efficiency Analysis:** Quantify potential electricity/resource consumption reduction from widespread adoption of optimal routing technologies.
* **Learning when to fail:** Sometimes no model will successfully answer a user query. Can we detect when we should fail cheaply?
* **Learning with uncertain signals:** How does judge noise affect the router training process? How does noisy feedback data affect the judge/router training process? Is off-policy data a problem when it comes to training routers?
* **Risk sensitivity:** Rather than optimizing for expected cost/quality, can we optimize for some other risk profile? E.g. we might tolerate a slightly higher cost and lower quality, if we reduce the variance or minimize a long tail.
* **Create a distilled predictor:** The *Language Models (Mostly) Know What They Know* paper shows that a model can sometimes predict whether it will be able to answer a question correctly. For a selected open “base” model, create a smaller “distilled predictor” that mirrors the base model’s ability to predict answer correctness (but can no longer calculate the answer). You might use the techniques from that paper and/or the pruning and distillation techniques from the movement pruning to shrink the predictor.

---

### Resources

* [The Martian API Quick Start Guide](#)
* The Martian Introductory Guide: [withmartian/martian-sdk-python: Martian Python SDK to manage judges and routers](https://github.com/withmartian/martian-sdk-python)

#### Background Reading

* [Apart x Martian Expert Orchestration blog](#)
* Recent research on cost-aware model routing (CARROT, HybridLLM, P2L)
* Papers on LLM evaluation techniques
* Research on ensemble methods and model merging
* Literature on democratizing AI development

#### Code & APIs

* [Loom video explaining Martian’s Router and Judge APIs (13 min video)](#)
* **[ROUTERBENCH: A Benchmark for Multi-LLM Routing System](#)**: This work formalizes and advances the development of LLM routing systems but also sets a standard for their assessment, paving the way for more accessible and economically viable LLM deployments.
* [The code for the paper ROUTERBENCH: A Benchmark for Multi-LLM Routing System](#)
* **[RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing](#)**: A principled framework for LLM routing based on preference data.

#### Available Tools

* Model evaluation frameworks
* Routing algorithm templates
* Martian APIs for creating judges, routers and running routers across models.
* Benchmarking datasets for testing router and judge performance

#### Some Ideas to Get You Started

* **Mechanistic Analysis of Manipulation Vulnerabilities in AI Orchestration:** Investigate the security vulnerabilities of judge and router systems through systematic adversarial testing, focusing on whether malicious inputs can manipulate routing decisions or compromise evaluation accuracy in Expert Orchestration frameworks.
* **Mechanistic Analysis of Mathematical Computation Pathways in Neural Networks:** Investigate how different mathematical algorithms are represented as vectors in activation space, testing the hypothesis that distinct algorithms correspond to orthogonal or near-orthogonal vectors. Explore whether unknown algorithm vectors can be discovered and activated through mechanistic intervention.
* **Building Interpretable Evaluation Systems for Helpfulness, Confidence, and Sycophancy Detection:** Develop sophisticated judge models that simultaneously evaluate AI responses across multiple critical dimensions using mechanistic interpretability techniques. The system combines whitebox methods (analyzing internal model states) with blackbox approaches (log probability analysis) to create comprehensive evaluation frameworks.
* **Distilling Knowledge Evaluation from Large Language Models:** Create ultra-lightweight "predictor" models that can determine whether a larger base model can answer a specific query, without having the computational ability to calculate the answer themselves. This involves mechanistic interpretability techniques to understand how models represent their own knowledge boundaries.
* **Building Interpretable Judge Models for Mathematical Reasoning Routes:** Develop a system that uses mechanistic interpretability to detect exactly when and how language models choose between different algorithms (addition, subtraction, multiplication) based on query structure. The project involves creating test datasets where key words determine algorithm selection, then using sparse autoencoders and activation analysis to identify the decision points in model processing.


