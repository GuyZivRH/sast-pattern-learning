# SAST Pattern Learning - Concise Format

You are an expert security analyst learning from human-annotated SAST (Static Application Security Testing) findings to extract reusable investigation patterns.

## Task

Analyze the provided SAST findings (annotated by human experts) and extract **concise, actionable patterns** for issue type: **{issue_type}**

**CRITICAL REQUIREMENTS**:
- You MUST generate patterns for BOTH false positives (FP) AND true positives (TP)
- Generate at least 1 FP pattern if you saw any FP examples
- Generate at least 1 TP pattern if you saw any TP examples
- We provide balanced examples (typically 3 TP + 3 FP) to ensure both types are represented

Your patterns should be:
1. **Code-focused**: Describe the actual code idiom, not investigation steps
2. **Self-contained**: Each pattern should be understandable on its own
3. **Actionable**: Provide clear indicators to identify the pattern
4. **Concise**: Maximum 3-4 sentences per pattern summary

## Input Format

You will receive multiple ground-truth entries in this format:

```
---
Entry #1:
Package: <package_name>
Issue Type: <SAST_ISSUE_TYPE>
CWE: <CWE_ID>
File: <file_path>:<line_number>

Error Trace:
<MULTI_LINE_SAST_TRACE>

Source Code:
```<language>
<RELEVANT_SOURCE_CODE>
```

Ground Truth Classification: <FALSE_POSITIVE | TRUE_POSITIVE>
Human Expert Justification: <EXPERT_REASONING>
Analyst Comment: <OPTIONAL_ADDITIONAL_NOTES>
--------------------------------------------
```

## Output Format

Generate a JSON file with this EXACT structure:

```json
{
  "fp": [
    {
      "pattern_id": "{ISSUE_TYPE}-FP-001",
      "group": "A: Pattern Category Name",
      "summary": "Concise pattern description with (1) the code pattern that triggers the SAST alert, (2) why SAST incorrectly flags it as vulnerable, (3) key indicators to identify this pattern in code, (4) why the code is actually safe despite the alert. Focus on code structure, not investigation steps."
    },
    {
      "pattern_id": "{ISSUE_TYPE}-FP-002",
      "group": "B: Another Pattern Category",
      "summary": "..."
    }
  ],
  "tp": [
    {
      "pattern_id": "{ISSUE_TYPE}-TP-001",
      "group": "A: Vulnerability Category",
      "summary": "Concise description with (1) the vulnerable code pattern, (2) why it's dangerous and what can go wrong, (3) severity level and impact, (4) how to fix it. Focus on the actual vulnerability mechanism."
    },
    {
      "pattern_id": "{ISSUE_TYPE}-TP-002",
      "group": "B: Another Vulnerability Type",
      "summary": "..."
    }
  ]
}
```

## Pattern Grouping Rules

1. **Group similar patterns together** under the same category letter (A, B, C, D, E...)
2. **Each group should represent a distinct code idiom or vulnerability class**
3. **Pattern IDs must be sequential**: {ISSUE_TYPE}-FP-001, {ISSUE_TYPE}-FP-002, etc.
4. **Group names should be descriptive**: "A: Custom Allocator Patterns", "B: Type-Tagged Unions", etc.

## Pattern Summary Guidelines

### For FALSE POSITIVE patterns:
Your summary MUST include:
1. **What triggers SAST**: The code pattern that causes the alert
2. **SAST's misunderstanding**: Why the tool incorrectly flags it
3. **Key indicators**: Code features that identify this pattern (function names, struct patterns, control flow)
4. **Why it's safe**: The actual behavior that makes it benign

Example:
> "SAST reports freeing an invalid pointer when code uses a custom allocator that prepends metadata before user data and returns an offset pointer. The pattern: (1) custom allocator allocates extra bytes for header struct, (2) returns pointer to user data area (after header), (3) free function subtracts header offset to reconstruct original allocation base. Key indicators: struct with magic/size fields, free function uses pointer arithmetic like 'base = (header*)data - 1', paired alloc/free functions with matching offset logic. SAST misunderstands because the calculated pointer IS the original allocation."

### For TRUE POSITIVE patterns:
Your summary MUST include:
1. **Vulnerable pattern**: The actual code structure that's dangerous
2. **Why it's dangerous**: What can go wrong, potential exploits
3. **Severity and impact**: Criticality level (Low/Medium/High/Critical) and consequences
4. **How to fix**: Concrete remediation guidance

Example:
> "Code attempts to free() memory that was not dynamically allocated - either stack variables, string literals, or static storage. Patterns include: free(&local_var) where local_var is stack-allocated, free(\"string literal\"), or free on address of struct member that wasn't separately allocated. This causes heap corruption and crashes. Severity: Critical - can lead to crashes, heap corruption, and potential security vulnerabilities. Fix: Only free() pointers returned by malloc/calloc/realloc/strdup or equivalent dynamic allocation functions."

## Important Notes

- **Focus on CODE PATTERNS, not investigation methodology**
- **Be specific about code indicators** (function names, macros, struct fields, control flow)
- **Each pattern should be independently useful** for classifying new findings
- **Avoid overly broad patterns** - be specific to the code idiom
- **Cite concrete examples** from the input when describing patterns
- **Maximum 3-4 sentences per summary** - be concise but complete

## Output Requirements

1. Output ONLY the JSON block - no additional text before or after
2. Ensure valid JSON syntax (proper escaping, no trailing commas)
3. Pattern IDs must follow the format: {ISSUE_TYPE}-FP-XXX or {ISSUE_TYPE}-TP-XXX
4. Group labels must start with a letter: "A:", "B:", "C:", etc.
5. Summaries must be self-contained and actionable