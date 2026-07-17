---
name: planning-feature
description: Planning skill for new features - includes changelog summary and test case consideration
---

# Feature Planning

Use this skill when implementing NEW functionality or capabilities.

<IMPORTANT>
**Prerequisites:** Invoke `planning-base` skill first if not already loaded. It provides the core planning phases (0-5).

Follow all base planning rules, plus the feature-specific rules below.
</IMPORTANT>

## Feature-Specific Requirements

### Changelog Summary

Every new feature MUST include a changelog summary. After analysis, create a concise summary:

```markdown
## Changelog Entry

### Added
- <Brief description of the new feature>
- <User-facing benefit or capability>
```

This summary will be added to CHANGELOG.md upon completion.

### Test Plan

After COMPLETING the feature implementation:

1. **Create test plan file:** `planning/testplan-<feature-name>.md`
2. Invoke `testing-testplan` skill for the template
3. Document automated tests and manual verification scenarios
4. Include regression checks for related features

### Unit Tests

Then ASK the user:

> "Feature implementation is complete. Would you like me to add unit tests?
>
> - Which components/functions should be tested?
> - Are there specific edge cases to cover?"

If user requests tests:

1. Invoke the testing skill: `testing-gtest-gmock` (C++) or `testing-pytest` (Python)
2. Follow the test planning process from that skill
3. Write tests ONE BY ONE, waiting for user approval after each test
4. Update test plan file with test status

### Pull Request

After tests (if any), ASK the user:

> "Ready to create a Pull Request. Should I proceed?"

If yes, invoke `git-create-pull-request` skill and create PR with:

- **Motivation** - Why this feature is needed
- **Technical Details** - What was implemented
- **Test Plan** - How it was tested

## Plan File Format

Save to `planning/feature-<name>.md`:

```markdown
# Feature: <Feature Name>

## Goal
<Original user request>

## Changelog Summary
### Added
- <Feature description>

## Analysis
<Key findings, scope, dependencies, risks>

## PR Strategy
<Single PR or split? If split, list each PR with scope>

### PR 1: <Title> (if multiple PRs)
**Scope:** <What's included>
**Estimated size:** ~X lines

## Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
- [ ] Update CHANGELOG.md

## Test Cases
- [ ] <Test type>: <Test description>
(or "No tests requested")

## Notes
<Any additional context or decisions>
```

## Example

**User request:** "Add user profile avatar upload"

**Planning output:**

```markdown
# Feature: User Profile Avatar Upload

## Goal
Add ability for users to upload and display profile avatars

## Changelog Summary
### Added
- User profile avatar upload with image preview
- Support for JPG, PNG, and WebP formats up to 5MB

## Analysis
- Scope: ProfileSettings component, user API, storage service
- Dependencies: existing file upload utility, image processing library
- Risks: Large file handling, image format validation

## PR Strategy
Single PR (~350 lines) - scope is focused and reviewable.

## Tasks
- [ ] Add avatar upload endpoint to user API
- [ ] Create AvatarUpload component with preview
- [ ] Implement image validation and resizing
- [ ] Update ProfileSettings to include avatar section
- [ ] Add avatar display to user profile header
- [ ] Update CHANGELOG.md

## Test Cases
- [ ] Unit: Avatar validation (size, format)
- [ ] Unit: Image resizing function
- [ ] Integration: Upload flow end-to-end

## Notes
- Max file size: 5MB
- Supported formats: JPG, PNG, WebP
- Avatar stored in /uploads/avatars/
```
