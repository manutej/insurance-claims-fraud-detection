# Prompt Engineering Framework for Insurance Fraud Detection

## Overview

This framework defines the comprehensive prompt engineering strategy for the GenAI-based insurance fraud detection system, including templates, optimization techniques, and best practices for maximum accuracy and reliability.

## Core Prompt Templates

### 1. Master Fraud Analysis Template

```python
MASTER_FRAUD_ANALYSIS_PROMPT = """
<role>
You are an elite insurance fraud investigator with 20+ years of experience in healthcare claims analysis, medical coding, and fraud pattern recognition. You have successfully identified over $100M in fraudulent claims.
</role>

<context>
Current Date: {current_date}
Analysis Session ID: {session_id}
Regulatory Framework: {regulations}
Historical Fraud Rate: {baseline_fraud_rate}%
</context>

<task>
Perform a comprehensive fraud analysis of the following insurance claim using multi-dimensional evaluation criteria.
</task>

<claim_data>
{formatted_claim_data}
</claim_data>

<historical_context>
Provider History:
{provider_history}

Patient History:
{patient_history}

Similar Claims:
{similar_claims}
</historical_context>

<analysis_instructions>
1. INITIAL SCAN
   - Identify any immediate red flags or anomalies
   - Note any missing or suspicious data
   - Check claim completeness and consistency

2. PATTERN ANALYSIS
   For each fraud type, evaluate probability (0-100%):

   a) Upcoding:
      - Compare diagnosis severity with procedure complexity
      - Check if coding maximizes reimbursement inappropriately
      - Compare with typical coding for similar conditions

   b) Phantom Billing:
      - Verify service plausibility given provider schedule
      - Check for impossible service combinations
      - Validate against patient availability

   c) Unbundling:
      - Identify services that should be bundled
      - Calculate cost differential
      - Check for systematic unbundling patterns

   d) Duplicate Billing:
      - Search for similar recent claims
      - Check for re-submissions with modifications
      - Identify split billing attempts

   e) Kickback/Collusion:
      - Analyze referral patterns
      - Check for reciprocal arrangements
      - Identify network anomalies

3. CONTEXTUAL VALIDATION
   - Medical Necessity: Does treatment match diagnosis?
   - Timeline Consistency: Are service dates logical?
   - Geographic Feasibility: Are locations reasonable?
   - Provider Capability: Can provider perform these services?

4. COMPARATIVE ANALYSIS
   - Compare to peer providers (same specialty/region)
   - Compare to patient's typical utilization
   - Compare to clinical guidelines

5. RISK SCORING
   Calculate composite fraud risk incorporating:
   - Individual red flag weights
   - Pattern confidence levels
   - Historical accuracy adjustments
</analysis_instructions>

<output_format>
## FRAUD ANALYSIS REPORT

### Executive Summary
[2-3 sentence overview of findings]

### Risk Assessment
- Overall Fraud Risk: [MINIMAL|LOW|MODERATE|HIGH|CRITICAL]
- Confidence Level: [0-100%]
- Primary Concern: [Main fraud type if detected]

### Detailed Findings
#### Red Flags Identified
1. [Specific finding with evidence]
2. [Specific finding with evidence]
...

#### Fraud Type Analysis
- Upcoding: [Score]% - [Evidence]
- Phantom Billing: [Score]% - [Evidence]
- Unbundling: [Score]% - [Evidence]
- Duplicate: [Score]% - [Evidence]
- Kickback: [Score]% - [Evidence]

### Supporting Evidence
[Detailed evidence with data points]

### Recommendation
[APPROVE|REVIEW|INVESTIGATE|DENY]

### Investigation Priority
[If investigation recommended, specify focus areas]

### Confidence Factors
- Factors increasing confidence: []
- Factors decreasing confidence: []
- Data limitations: []
</output_format>

<important_notes>
- Maintain objectivity and avoid assumptions
- Cite specific data points for all conclusions
- Acknowledge uncertainty when present
- Consider legitimate explanations for anomalies
</important_notes>
"""
```

### 2. Medical Necessity Evaluation Prompt

```python
MEDICAL_NECESSITY_PROMPT = """
<role>
You are a board-certified physician with expertise in utilization review and medical necessity determination.
</role>

<clinical_context>
Diagnosis Codes: {diagnosis_codes}
Diagnosis Descriptions: {diagnosis_descriptions}
Procedure Codes: {procedure_codes}
Procedure Descriptions: {procedure_descriptions}
Patient Demographics: Age {age}, Gender {gender}
Relevant History: {medical_history}
</clinical_context>

<clinical_guidelines>
{relevant_clinical_guidelines}
</clinical_guidelines>

<evaluation_criteria>
1. INDICATION APPROPRIATENESS
   - Is the procedure indicated for the diagnosis?
   - Are there documented symptoms/findings supporting intervention?
   - Have conservative treatments been attempted when appropriate?

2. LEVEL OF SERVICE
   - Is the complexity appropriate for the condition?
   - Could the same outcome be achieved with less intensive service?
   - Is the setting (inpatient/outpatient) appropriate?

3. FREQUENCY AND DURATION
   - Is the frequency of service clinically justified?
   - Does duration align with standard treatment protocols?
   - Are there signs of overutilization?

4. CLINICAL EVIDENCE
   - Do the services follow evidence-based guidelines?
   - Are there peer-reviewed studies supporting this approach?
   - Does it align with specialty society recommendations?
</evaluation_criteria>

<output>
### Medical Necessity Determination

**Decision**: [MEDICALLY NECESSARY|POTENTIALLY UNNECESSARY|NOT MEDICALLY NECESSARY]

**Clinical Rationale**:
[Detailed medical justification]

**Guideline Alignment**:
- Follows Guidelines: [YES/NO/PARTIAL]
- Specific Guidelines: [List applicable guidelines]
- Deviations Noted: [Any departures from standard care]

**Alternative Treatment Considerations**:
[If service seems inappropriate, suggest alternatives]

**Confidence in Assessment**: [HIGH|MEDIUM|LOW]
**Reasoning**: [Explain confidence level]
</output>
"""
```

### 3. Provider Behavior Analysis Prompt

```python
PROVIDER_BEHAVIOR_PROMPT = """
<role>
You are a healthcare data analyst specializing in provider profiling and aberrant billing pattern detection.
</role>

<provider_profile>
Provider ID: {provider_id}
NPI: {provider_npi}
Specialty: {specialty}
Years in Practice: {years_practice}
Location: {location}
</provider_profile>

<billing_statistics>
Current Period Stats:
- Total Claims: {total_claims}
- Average Claim Amount: ${avg_amount}
- Procedure Mix: {procedure_distribution}
- Diagnosis Mix: {diagnosis_distribution}
- Complexity Distribution: {complexity_levels}

Peer Comparison (Same Specialty/Region):
- Peer Average Claim: ${peer_avg_amount}
- Peer Procedure Mix: {peer_procedure_distribution}
- Statistical Deviation: {standard_deviations}σ
</billing_statistics>

<historical_patterns>
{provider_history_summary}
</historical_patterns>

<analysis_requirements>
1. BILLING PATTERN ANALYSIS
   - Identify unusual billing concentrations
   - Detect sudden pattern changes
   - Find statistical outliers

2. PEER COMPARISON
   - Compare to specialty norms
   - Adjust for patient mix
   - Consider geographic factors

3. TEMPORAL ANALYSIS
   - Identify trending changes
   - Detect seasonal anomalies
   - Find day/time patterns

4. RED FLAG IDENTIFICATION
   - Consistent high-complexity billing
   - Unusual procedure combinations
   - Impossible service volumes

5. RISK STRATIFICATION
   - Calculate provider risk score
   - Identify specific risk factors
   - Recommend monitoring level
</analysis_requirements>

<output>
### Provider Behavior Analysis Report

**Provider Risk Tier**: [LOW|MODERATE|HIGH|CRITICAL]

**Key Findings**:
1. [Most significant finding]
2. [Second significant finding]
...

**Statistical Anomalies**:
- [Metric]: [Value] ([X]σ from mean)
...

**Peer Comparison Results**:
- Billing [Above/Below] peer average by [X]%
- Unusual patterns: [List specific patterns]

**Behavioral Indicators**:
✓/✗ Upcoding tendency: [Evidence]
✓/✗ Volume aberration: [Evidence]
✓/✗ Complexity inflation: [Evidence]
✓/✗ Service clustering: [Evidence]

**Recommendation**:
[STANDARD MONITORING|ENHANCED MONITORING|IMMEDIATE INVESTIGATION|PREPAYMENT REVIEW]

**Confidence**: [0-100]%
</output>
"""
```

### 4. Network Analysis Prompt

```python
NETWORK_FRAUD_PROMPT = """
<role>
You are a forensic data scientist specializing in network analysis and organized healthcare fraud detection.
</role>

<network_context>
Primary Provider: {provider_id}
Network Connections:
{network_graph_description}

Referral Patterns:
- Outgoing Referrals: {outgoing_referrals}
- Incoming Referrals: {incoming_referrals}
- Reciprocal Rate: {reciprocal_percentage}%
</network_context>

<suspicious_patterns>
Evaluate for:
1. COLLUSION INDICATORS
   - Closed referral loops
   - Unusual reciprocity rates
   - Geographic anomalies in referrals

2. KICKBACK SCHEMES
   - Concentrated referral patterns (>80% to single provider)
   - Financial relationships between entities
   - Sudden referral pattern changes

3. PATIENT STEERING
   - Unnecessary referral chains
   - Bypassing closer/appropriate providers
   - Pattern of maximum billing through network

4. ORGANIZED FRAUD RINGS
   - Multiple providers with identical patterns
   - Synchronized billing spikes
   - Shared patient populations with no geographic logic
</suspicious_patterns>

<graph_metrics>
Centrality Scores:
- Degree Centrality: {degree_centrality}
- Betweenness Centrality: {betweenness_centrality}
- PageRank: {pagerank_score}

Clustering Coefficient: {clustering_coefficient}
Network Density: {network_density}
</graph_metrics>

<output>
### Network Fraud Analysis

**Network Risk Level**: [NORMAL|SUSPICIOUS|HIGH RISK|CONFIRMED FRAUD RING]

**Collusion Probability**: [0-100]%

**Key Network Anomalies**:
1. [Specific anomaly with evidence]
2. [Specific anomaly with evidence]

**Suspicious Relationships Identified**:
- [Provider A] ↔ [Provider B]: [Nature of suspicion]

**Pattern Classification**:
□ Legitimate referral network
□ Possible kickback arrangement
□ Likely collusion ring
□ Organized fraud operation

**Investigation Recommendations**:
- Priority Targets: [List providers to investigate]
- Evidence to Collect: [Specific data needed]
- Surveillance Period: [Recommended duration]

**Network Visualization Notes**:
[Description of suspicious network patterns for visualization]
</output>
"""
```

## Prompt Optimization Strategies

### 1. Dynamic Few-Shot Learning

```python
class DynamicFewShotSelector:
    """
    Selects most relevant examples based on current claim
    """

    def select_examples(self, claim, example_bank, n_examples=5):
        # Calculate similarity scores
        similarities = []
        for example in example_bank:
            score = self.calculate_similarity(claim, example)
            similarities.append((score, example))

        # Select top N most similar examples
        top_examples = sorted(similarities, key=lambda x: x[0], reverse=True)[:n_examples]

        # Format for prompt inclusion
        formatted_examples = self.format_examples(top_examples)

        return formatted_examples

    def calculate_similarity(self, claim, example):
        """
        Multi-factor similarity calculation
        """
        weights = {
            'diagnosis_match': 0.3,
            'procedure_match': 0.3,
            'amount_similarity': 0.1,
            'provider_type_match': 0.1,
            'fraud_type_match': 0.2
        }

        similarity_score = 0
        for factor, weight in weights.items():
            similarity_score += self.compare_factor(claim, example, factor) * weight

        return similarity_score
```

### 2. Chain-of-Thought Optimization

```python
COT_OPTIMIZATION_TEMPLATE = """
Let's analyze this claim step by step to ensure accuracy:

Step 1: Data Validation
- First, I'll verify all claim fields are complete and valid...
- [Validation results]

Step 2: Initial Risk Assessment
- Now, I'll calculate the baseline risk score...
- [Risk calculation]

Step 3: Pattern Matching
- Next, I'll compare against known fraud patterns...
- [Pattern analysis]

Step 4: Contextual Analysis
- I'll examine the broader context of this claim...
- [Context evaluation]

Step 5: Evidence Synthesis
- Combining all findings...
- [Synthesis]

Step 6: Confidence Calibration
- Adjusting confidence based on data quality and completeness...
- [Final confidence]

Therefore, my conclusion is: [Final determination with reasoning]
"""
```

### 3. Self-Consistency Verification

```python
class SelfConsistencyChecker:
    """
    Runs multiple variations and checks for consistency
    """

    def verify_consistency(self, claim, base_prompt, n_iterations=3):
        results = []

        # Run with different temperatures
        temperatures = [0.3, 0.5, 0.7]

        for temp in temperatures:
            response = self.llm.generate(
                prompt=base_prompt.format(claim=claim),
                temperature=temp
            )
            results.append(self.parse_result(response))

        # Check consistency
        consistency_score = self.calculate_consistency(results)

        if consistency_score < 0.8:
            # Run with more detailed prompt if inconsistent
            detailed_prompt = self.add_clarification(base_prompt)
            return self.llm.generate(detailed_prompt.format(claim=claim), temperature=0.3)

        # Return majority vote result
        return self.majority_vote(results)
```

## Prompt Security & Safety

### 1. Input Sanitization

```python
class PromptSanitizer:
    """
    Sanitizes inputs to prevent prompt injection
    """

    def sanitize(self, user_input):
        # Remove potential injection attempts
        sanitized = user_input

        # Remove system-like commands
        injection_patterns = [
            r"ignore (previous|all) instructions",
            r"system\s*:",
            r"assistant\s*:",
            r"<\|.*?\|>",  # Special tokens
            r"```system",
            r"SYSTEM PROMPT"
        ]

        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)

        # Escape special characters
        sanitized = self.escape_special_chars(sanitized)

        # Validate against schema
        self.validate_schema(sanitized)

        return sanitized
```

### 2. Output Validation

```python
class OutputValidator:
    """
    Validates LLM outputs for safety and accuracy
    """

    def validate(self, output, expected_format):
        checks = {
            'format_compliance': self.check_format(output, expected_format),
            'no_hallucination': self.check_hallucination(output),
            'no_pii_leakage': self.check_pii(output),
            'factual_consistency': self.check_facts(output),
            'confidence_calibration': self.check_confidence(output)
        }

        if not all(checks.values()):
            raise ValidationError(f"Output validation failed: {checks}")

        return output
```

## Performance Optimization

### 1. Prompt Compression

```python
class PromptCompressor:
    """
    Compresses prompts to reduce token usage
    """

    def compress(self, prompt):
        # Remove redundancy
        prompt = self.remove_redundancy(prompt)

        # Use abbreviations for common terms
        abbreviations = {
            "diagnosis": "dx",
            "procedure": "px",
            "provider": "prv",
            "patient": "pt",
            "fraud_indicator": "fi"
        }

        for full, abbr in abbreviations.items():
            prompt = prompt.replace(full, abbr)

        # Compress structured data
        prompt = self.compress_json(prompt)

        # Remove unnecessary whitespace
        prompt = ' '.join(prompt.split())

        return prompt
```

### 2. Caching Strategy

```python
class PromptCache:
    """
    Caches prompt results for similar queries
    """

    def __init__(self):
        self.semantic_cache = {}
        self.embedding_model = EmbeddingModel()
        self.similarity_threshold = 0.95

    def get_or_generate(self, prompt, generator_func):
        # Generate embedding for prompt
        embedding = self.embedding_model.encode(prompt)

        # Check cache for similar prompts
        for cached_embedding, cached_result in self.semantic_cache.items():
            similarity = cosine_similarity(embedding, cached_embedding)
            if similarity > self.similarity_threshold:
                return cached_result

        # Generate new result
        result = generator_func(prompt)

        # Cache the result
        self.semantic_cache[embedding] = result

        return result
```

## Advanced Techniques

### 1. Meta-Prompting

```python
META_PROMPT_TEMPLATE = """
You are a prompt engineering expert. Your task is to optimize the following prompt for fraud detection:

Original Prompt:
{original_prompt}

Optimization Goals:
1. Increase accuracy of fraud detection
2. Reduce false positives
3. Improve explanation clarity
4. Minimize token usage

Please provide:
1. Optimized prompt version
2. Explanation of changes
3. Expected improvement metrics
"""
```

### 2. Prompt Ensemble

```python
class PromptEnsemble:
    """
    Uses multiple prompt variations and combines results
    """

    def __init__(self):
        self.prompt_variants = [
            self.analytical_prompt,
            self.investigative_prompt,
            self.medical_prompt,
            self.statistical_prompt
        ]

    def analyze(self, claim):
        results = []
        weights = [0.3, 0.3, 0.2, 0.2]  # Weight for each prompt type

        for prompt_func, weight in zip(self.prompt_variants, weights):
            result = prompt_func(claim)
            results.append((result, weight))

        # Weighted combination of results
        final_result = self.combine_weighted(results)

        return final_result
```

### 3. Adaptive Prompting

```python
class AdaptivePromptSystem:
    """
    Adapts prompts based on performance feedback
    """

    def __init__(self):
        self.performance_history = []
        self.prompt_variations = {}
        self.optimizer = PromptOptimizer()

    def adapt_prompt(self, base_prompt, performance_metrics):
        # Analyze recent performance
        recent_performance = self.performance_history[-100:]

        # Identify weak areas
        weak_areas = self.identify_weaknesses(recent_performance)

        # Modify prompt to address weaknesses
        if 'high_false_positive' in weak_areas:
            base_prompt = self.add_specificity(base_prompt)

        if 'low_detection_rate' in weak_areas:
            base_prompt = self.broaden_detection(base_prompt)

        if 'poor_explanation' in weak_areas:
            base_prompt = self.enhance_explanation(base_prompt)

        return base_prompt
```

## Prompt Testing Framework

```python
class PromptTestSuite:
    """
    Comprehensive testing for prompt effectiveness
    """

    def __init__(self):
        self.test_cases = self.load_test_cases()
        self.metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'explanation_quality': 0,
            'token_efficiency': 0
        }

    def test_prompt(self, prompt_template):
        results = []

        for test_case in self.test_cases:
            # Run prompt
            output = self.run_prompt(prompt_template, test_case['input'])

            # Evaluate output
            evaluation = self.evaluate_output(
                output,
                test_case['expected_output'],
                test_case['ground_truth']
            )

            results.append(evaluation)

        # Calculate metrics
        self.metrics = self.calculate_metrics(results)

        return self.generate_report()

    def evaluate_output(self, output, expected, ground_truth):
        return {
            'correct_classification': output['fraud_type'] == ground_truth['fraud_type'],
            'score_accuracy': abs(output['score'] - ground_truth['score']) < 0.1,
            'explanation_quality': self.rate_explanation(output['explanation']),
            'token_count': self.count_tokens(output),
            'response_time': output['processing_time']
        }
```

## Best Practices & Guidelines

### 1. Prompt Structure Guidelines

- **Clear Role Definition**: Always start with explicit role definition
- **Structured Context**: Provide context in organized sections
- **Specific Instructions**: Use numbered steps for complex analysis
- **Output Format**: Define exact output structure required
- **Examples When Needed**: Include 2-3 relevant examples for complex tasks

### 2. Token Optimization Rules

- **Abbreviate Common Terms**: Use consistent abbreviations
- **Remove Redundancy**: Eliminate repeated information
- **Use References**: Reference previous context instead of repeating
- **Compress Data**: Use efficient data formats (arrays vs. verbose descriptions)

### 3. Reliability Techniques

- **Temperature Control**: Use low temperature (0.3-0.5) for factual analysis
- **Self-Consistency**: Run multiple times and check agreement
- **Confidence Calibration**: Always request confidence scores
- **Fallback Prompts**: Have simpler prompts ready if complex ones fail

### 4. Safety Considerations

- **Input Validation**: Always validate and sanitize inputs
- **Output Verification**: Check outputs against expected schema
- **PII Protection**: Mask sensitive information in prompts
- **Audit Logging**: Log all prompts and responses for compliance

## Continuous Improvement Process

```python
class PromptImprovementPipeline:
    """
    Continuously improves prompts based on feedback
    """

    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.prompt_analyzer = PromptAnalyzer()
        self.ab_tester = ABTester()

    def improvement_cycle(self):
        while True:
            # Collect feedback
            feedback = self.feedback_collector.get_recent_feedback()

            # Analyze prompt performance
            performance = self.prompt_analyzer.analyze(feedback)

            # Generate prompt variations
            variations = self.generate_variations(performance)

            # A/B test variations
            winning_prompt = self.ab_tester.test(variations)

            # Deploy winning prompt
            self.deploy(winning_prompt)

            # Wait for next cycle
            time.sleep(86400)  # Daily improvement cycle
```

## Conclusion

This prompt engineering framework provides a comprehensive approach to leveraging LLMs for insurance fraud detection. By combining structured templates, optimization techniques, and continuous improvement processes, we can achieve high accuracy while maintaining explainability and compliance requirements.

The framework is designed to be:
- **Modular**: Components can be used independently
- **Scalable**: Handles high-volume processing
- **Adaptive**: Learns and improves over time
- **Secure**: Protected against prompt injection
- **Efficient**: Optimized for token usage and cost