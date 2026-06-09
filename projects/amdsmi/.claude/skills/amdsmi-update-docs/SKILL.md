---
name: amdsmi-update-docs
description: "Create and update AMD SMI documentation. Use when: writing docs, adding a new page, updating API reference, fixing doc content, wiring toctree."
---

# Update AMD SMI docs

Creates and updates documentation for AMD SMI.

## When to use

- Adding a new how-to, conceptual, install, or reference page
- Updating existing docs after an API change or CLI change
- Wiring a new page into `_toc.yml.in` for discoverability
- Adding or updating Doxygen comments in `amdsmi.h`
- Fixing style issues

## Guidelines

### Style preferences

Apply these on every page you create or touch:

- [Google developer documentation style guide](https://developers.google.com/style)

  Rules include:

  - **Sentence-case headings**: capitalize only the first word and proper nouns.
  - **Parallel structures**: list parallel concepts in a grammatically and stylistically parallel way.
  - **Front-load**: lead with the most important idea.
  - **Accessibility**: use inclusive language.

- **Markdown**
  - Most `.md` files under `docs/` use [MyST-flavored Markdown](https://myst-parser.readthedocs.io/en/latest/) through [Sphinx](https://www.sphinx-doc.org/en/master/usage/markdown.html).
  - For `.md` files not intended to be processed by Sphinx, like READMEs typically read in GitHub previews, use [GitHub-flavored Markdown](https://github.github.com/gfm/).
  - Use [Pygments](https://pygments.org/languages/) language identifiers on all fenced code blocks.

#### Content architecture

Follow [Diátaxis](https://diataxis.fr/) best practices when structuring documentation.
This approach identifies four needs that users have - tutorials, how-to guides,
technical reference and conceptual explanation.

| Content type                          | Directory     | Ask:                                        |
|:--------------------------------------|:--------------|:--------------------------------------------|
| Install, uninstall, build steps       | `install/`    | "How do I set it up?"                       |
| Step-by-step recommended usage        | `how-to/`     | "How do I do X?"                            |
| API reference, CLI reference, options | `reference/`  | "What does this do / what are the options?" |
| Background, concepts, design          | `conceptual/` | "Why does this work this way?"              |

When unsure, prefer `how-to/` for procedural content and `conceptual/` for explanatory content.

#### Page template

Use a logical structure for new pages. Adjust sections to fit the content type. For example:

```markdown
---
myst:
  html_meta:
    "description lang=en": "<one-sentence description>"
    "keywords": "amdsmi, <topic keywords that don't already appear on the page>"
---

# <Sentence-case title>

<One-paragraph orientation: what this page covers and who it is for.>

## Prerequisites

<List of requirements before the user starts.>

## <Main sections>

<Content.>

## Further reading

- [<Title>](<path>)
```

### Cross-references and linking

Use these directives to link to API constructs auto-documented by Doxygen. If
a public API name changes, update all documentation references to it to use the
new name.

- Use [Sphinx C domain references](https://www.sphinx-doc.org/en/master/usage/domains/c.html#the-c-domain) to refer to APIs.

  ```
  {c:func}`amdsmi_init`
  {c:type}`amdsmi_status_t`
  ```

  You can also refer to groups of APIs categorized using the `@defgroup` Doxygen marker in `amdsmi.h`.

  ```
  {ref}`anchor text <tagInitShutdown>`
  ```

- Use normal Markdown references to refer to Python APIs in `docs/reference/amdsmi-py-api.md`.

## Process

1. **Read first** — read the target file before editing; read a peer page for structure reference when creating.
2. **Create or update** the file(s) needed.
3. **Wire into toc** — add or update entries in `docs/sphinx/_toc.yml.in` and `docs/index.md` for any new pages.
4. **Apply style rules** — sentence-case headings, Pygments code block identifiers, correct naming conventions.
5. **Report** what was created/changed and what requires a Sphinx build to verify.

## Output

After completing the work, report:

```
## Docs update summary

**Created:** [file list]
**Modified:** [file list]
**Toc wired:** yes / no — [entry added]
**Open items:** [anything the user must supply, e.g., missing content]
```
