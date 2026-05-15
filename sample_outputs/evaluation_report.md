# Evaluation Report

## Retrieval
- Mean term recall@5: 0.67
- Mean source recall@5: 1.00
- Mean reciprocal rank: 0.88

## Extraction
- Documents processed: 4
- Blocks extracted: 141
- Field counts: {'parties': 6, 'dates': 19, 'addresses': 9, 'amounts': 11, 'signatures': 9, 'deadlines': 6, 'unclear_spans': 0}
- Extraction warnings: 11

## Draft Grounding
- Citation validation passed: True
- Invalid citations: []
- Uncited factual lines: 0
- Weakly supported cited lines: 0

## Edit Learning
- Feedback captured: True
- Stored feedback count: 1
- Learned: Carry operator-added cited facts into future drafts when similar evidence appears.
