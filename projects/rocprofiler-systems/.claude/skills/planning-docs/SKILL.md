---
name: planning-docs
description: Planning skill for documentation - no changelog updates
---

# Documentation Planning

Use this skill when creating or updating documentation.

<IMPORTANT>
**Prerequisites:** Invoke `planning-base` skill first if not already loaded. It provides the core planning phases (0-5).

Follow all base planning rules, plus the documentation-specific rules below.
Documentation changes do NOT require changelog updates.
</IMPORTANT>

## Documentation-Specific Requirements

### Documentation Type

Identify what type of documentation is being created/updated:

- **API Documentation** - Endpoint descriptions, parameters, responses
- **User Guide** - How-to instructions for end users
- **Developer Guide** - Setup, architecture, contribution guidelines
- **Code Comments** - Inline documentation, JSDoc/Doxygen
- **README** - Project overview, quick start
- **Architecture** - System design, data flow, diagrams

### Audience Analysis

Consider who will read this documentation:

1. **Who is the audience?** - Developers, users, administrators?
2. **What do they need to know?** - Concepts, procedures, references?
3. **What is their skill level?** - Beginner, intermediate, expert?

### Content Structure

Plan the documentation structure before writing:

- Use clear headings and hierarchy
- Include code examples where helpful
- Add diagrams for complex concepts
- Provide links to related documentation

## Plan File Format

Save to `planning/docs-<name>.md`:

```markdown
# Documentation: <Doc Name>

## Goal
<What documentation is being created/updated>

## Type
<API/User Guide/Developer Guide/Code Comments/README/Architecture>

## Audience
<Who will read this, their skill level, what they need>

## Analysis
<Current state, gaps, related docs>

## Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Review for clarity

## Structure
<Outline of documentation sections>

## Notes
<Style guides, templates to follow, related links>
```

## Example

**User request:** "Document the authentication flow"

**Planning output:**

```markdown
# Documentation: Authentication Flow

## Goal
Create comprehensive documentation of the authentication system

## Type
Developer Guide + Architecture

## Audience
- Backend developers integrating with auth system
- New team members understanding the codebase
- Skill level: Intermediate developers

## Analysis
- Current state: Only inline code comments exist
- Gaps: No flow diagrams, no token lifecycle explanation
- Related: API docs mention auth but don't explain it

## Tasks
- [ ] Create auth flow diagram (login, refresh, logout)
- [ ] Document token lifecycle and storage
- [ ] Explain role-based access control
- [ ] Add code examples for common auth operations
- [ ] Review and link from main README

## Structure
1. Overview
2. Authentication Flow Diagram
3. Token Management
   - Access tokens
   - Refresh tokens
   - Token storage
4. Role-Based Access Control
5. Code Examples
6. Troubleshooting

## Notes
- Use Mermaid for diagrams
- Follow existing docs/STYLE_GUIDE.md
- Link to API docs for endpoint details
```
