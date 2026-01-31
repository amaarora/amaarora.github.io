---
name: infographic-generator
description: Generate blog post infographics using a consistent visual style via image generation
allowed-tools: Read, Glob
---

## Infographic Generation for Blog Posts

This skill contains the prompt template used to generate infographics for Aman's blog posts. All infographics in the blog use this consistent style - hand-drawn editorial illustrations with a muted ink-and-paper aesthetic.

### How to use

1. Read the blog post to extract the title and section headers (or the specific section content if generating a section-level infographic)
2. Fill in the TITLE and SECTION HEADERS at the bottom of the prompt template
3. Send the complete prompt to an image generation model (Nano Banana Pro, 9:16, 1K resolution)

### Prompt Template

```
Create a 9:16 infographic (1K resolution) with Nano Banana Pro using the text that I'll share with you next. Do it in two steps:

STEP 1: Parse the text to extract the title and the headers
First, parse the text to extract the title and the section headers. I want you to use the EXACT headlines of the post to create the infographic. IMPORTANT: Do not include any other text.

STEP 2: Apply the following style guide

COLORS
Primary: #2C3E50 (deep ink blue)
Secondary: #8B9298 (pewter), #D4D1CC (warm gray)
Background: #FAF9F7 (warm white) | Text: #1C1C1C (charcoal)

TEXT
Title: Fraunces, serif, 48-72pt, #1C1C1C. Include a hand-drawn imperfect circle (#2C3E50) around the most important 1-2 words that's 3-5px stroke and slightly uneven.
Section headers: Space Grotesk Bold, sans serif, 14-24pt, #1C1C1C.
Use ONLY the post's exact headlines—do not include other text

LAYOUT
Grid-based structure with clear sections
Generous whitespace between elements
Subtle numbered badges (1-6) for each section using deep ink blue

ILLUSTRATION
Style: Loose architectural sketch meets editorial illustration. Think blueprint-inspired linework with organic, hand-drawn imperfections. Visible pencil/ink texture with intentional weight variation. NOT clean vector art.
Texture: Subtle crosshatching and stippling for shading. Paper grain visible. NO smooth gradients.
Proportions: Grounded and realistic, NOT cartoonish. Figures should feel like New Yorker or HBR editorial illustrations.
Concept: Each illustration must be a visual metaphor that deepens the headline's meaning—not a literal depiction. Avoid generic business imagery (NO robots, abstract icons, or clip art).
Palette: Ink blue, pewter, warm grays, and charcoal. Keep illustrations tonal and muted.
One illustration per headline

//
TITLE: [Your title here]
SECTION HEADERS:
[Header 1]
[Header 2]
[Header 3]
...
```

### Style Notes

- The style is consistent across all blog infographics - do not modify the colors, fonts, or illustration style
- For section-level infographics (e.g., evaluation framework, iteration cycle), adapt the layout but keep the same visual language
- The "hand-drawn imperfect circle" around key title words is a signature element
- Numbered badges correspond to section order in the post
- Resolution is always 1K, aspect ratio 9:16 for full-post infographics (section-level may vary)
