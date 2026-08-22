# AGENTS.md

Guidance for agents working in this repository.

## Project Context

This repository contains the CAS ETH innovation project "Regularized GLM--Neural Network Hybrids for Insurance Pricing".
The project compares GLM, neural-network, and GLM-neural-network hybrid approaches for insurance claim frequency modelling, with emphasis on the trade-off between predictive performance and GLM-like structure.

## Editing Principles

- Keep changes small and directly tied to the requested task.
- Preserve existing wording unless the request is specifically editorial.
- Prefer relative links for repository-local documents.
- Do not commit generated outputs such as PDFs, LaTeX auxiliary files, or locally downloaded literature PDFs.
- Check `.gitignore` before adding generated or downloaded files.

## Report Style

These conventions apply to both LaTeX and Markdown deliverables.

- Use a clear, academic tone: concise, precise, and matter-of-fact.
- Use British English spelling and phrasing, following ETH Zurich style: prefer `-yse` and `-ise` forms such as `analyse` and `theorise`.
- Prefer simple sentences and active verbs where possible.
- Use inclusive, respectful language; avoid gendered generic terms and use plural constructions or singular `they` where this improves neutrality.
- Write `ETH Zurich` or `ETH`, without an article; use `the` only when the name functions adjectivally, as in `the ETH Zurich Department of Mathematics`.
- Keep terminology consistent with the project outline: GLM, CANN, FNN, SHAP, Poisson deviance, and regularization.
- Avoid marketing-style language, overstatement, and unsupported claims.
- Use "regularization" for the modelling concept, but follow source titles exactly when quoting or citing them.
- Keep mathematical notation close to the outline unless there is a clear reason to revise it.
- Do not introduce new abbreviations unless they are used repeatedly and defined on first use.
- Introduce acronyms and initialisms in round brackets after the full term on first use.
- Use words for numbers from one to ten in running prose, and numerals for 11 and above.
- Use numerals with units, times, percentages, currencies, ranges, and mixed-number comparisons.
- In running text, write dates as `30 November 2015`; use compact numeric dates only in tables where space matters.
- Write out `percent` in running text where possible; use `%` mainly in tables, charts, formulas, and compact technical notation.
- Use sentence-style capitalisation for headings: capitalise only the first word and proper nouns.
- Use double quotation marks for quotations; use single quotation marks only for quotations inside quotations.
- Avoid mandatory Oxford commas; add the final comma only where needed for clarity.
- Use en dashes for ranges and hyphens for compound modifiers before nouns.
- Italicise titles of books and long works; put journal articles, book chapters, poems, and songs in quotation marks.

## LaTeX Documents

- Keep `deliverables/report/report.tex` minimal unless the user asks for additional formatting.
- Use `article` with `a4paper` and `babel` configured for British English.
- Prefer default LaTeX styling over custom title, heading, margin, or font formatting.
- Put prose in short, readable lines to keep diffs easy to review.

## Markdown Documents

- Use ATX headings (`#`, `##`, `###`).
- Keep README content brief and orienting; put longer project details in dedicated files.
- Use numbered lists for reference lists and bullet lists for short operational notes.
- Use repository-relative Markdown links, for example `[literature.md](literature.md)`.
