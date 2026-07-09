# General guidelines:
- Use AI tools as an instrument but avoid producing HLD via AI tools, especially for complex problems.
```
"Writing is thinking. To write well is to think clearly. That's why it's so hard." 
                                                        - David McCullough
```
- Concentrate on articulating why, ex "why is it a problem?", "why is this decision made?", etc.
- Be short and specific:
    - Avoid word-fluff which artificially increases text size. 
    - Avoid inventing new fancy terms, use simple wording.
    - Keep relevant information and decisions localized and structured - don't spread your thoughts over the whole document, organize them.
    - Use visual diagrams to provide more clarity.
- Try to strengthen your arguments with concrete data (estimations, measurements, etc.).
- Try to be specific where possible - concrete examples of problems being solved, measurements and estimations.
- Avoid adding information which becomes outdated fast, ex code locations, variable and class names are changing rather fast.
- Formatting matters - use bullet lists, tables, indentation and break down on paragraphs to accent the points. Don't use formatting where it just adds a visual noise.
- Outlining how the system currently works ("System Context") and what problem you aim to solve ("Problem Statement" and "Requirements") is the most important part. Quality of the design directly depends on how well you articulate what problems you are solving and why.

# Template
## System Context
- What part of the system which is in scope of this document currently does at a high level:
    - What purpose does it serve and what class of problem it currently solves? What it doesn't cover?
    - What are main components and relevant technical details?
    - What are surrounding components?
- Other relevant context which may affect the design:
    - External customer requests and feedback.
    - Competitors solutions.
    - Assumptions and constraints.

## Problem statement
- What problems this design tries to solve?
- Why these problems are important and what are their impact?

## Requirements
- Functional requirements - what system shall do. If list is very large a good idea is to prioritize these requirements.
- Non-functional requirements (performance, scalability, availability, security, etc).
- Guidelines which design needs to follow and justification.

## Design
- Describe proposed high-level design.
- Clearly articulate architectural decisions, why you've made them and what are alternatives.

## Implementation phases
- Describe in what phases the designed feature could be developed.
- Each phase should be defined as a "vertical slice", meaning it touches different part of the architecture stack and delivers an incremental value to the user.

## Validation, security and debuggability
- Outline required types of tests (unit tests, functional, integration) and validation strategy.
- Define relevant debugging capabilities to reduce maintenance cost such as logging, metrics, alerting, tracing strategy

## Open questions
- Known unknowns, deferred decisions, trade-offs.
