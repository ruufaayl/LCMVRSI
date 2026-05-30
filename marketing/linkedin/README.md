# LinkedIn post series — LCMVRSI

Ten ready-to-post `.md` files telling the project story in chronological order (foundation →
baselines → frontier → theory → mechanism → wrap-up). Each post is self-contained: hook, what was
done, how to use it, the GitHub link, SEO hashtags, and **which image to attach**.

**Repo:** https://github.com/ruufaayl/LCMVRSI
**Images:** in [`images/`](images/) next to this file.

## Posting schedule & image map

| # | Post | Theme | Image to attach |
|---|------|-------|-----------------|
| 1 | `post-01-kickoff.md` | Why this exists (the long-context recall problem) | `images/state_spectrum.png` |
| 2 | `post-02-foundation.md` | Rigor: tested harness, honesty discipline, CI | *(optional: screenshot of `uv run pytest` — 109 passing)* |
| 3 | `post-03-recall-wall.md` | The recall wall reproduced (MQAR) | `images/recall_vs_pairs.png` |
| 4 | `post-04-architectures.md` | Six architectures on one interface | `images/state_spectrum.png` |
| 5 | `post-05-frontier.md` | The recall–memory frontier + the key insight | `images/frontier_mqar.png` |
| 6 | `post-06-theorem.md` | H1: a proven entropy floor | *(quote card of the inequality — text in the post)* |
| 7 | `post-07-mechanism.md` | H2: the surprise-gated memory idea | *(optional: reuse `images/state_spectrum.png`)* |
| 8 | `post-08-negative-result.md` | The honest negative result | `images/h2_structured_recall.png` |
| 9 | `post-09-reproducibility.md` | Engineering: tests, dashboard, CI-built paper | *(optional: screenshot of the Streamlit dashboard)* |
| 10 | `post-10-recap.md` | Recap + what's next + call to action | `images/frontier_mqar.png` |

**Tips.** Posts 1, 3, 5, 8, 10 have strong standalone figures — lead with those for reach. Posts 2,
6, 7, 9 work as text or simple quote/screenshot cards. Space them ~2–3 days apart; reply to your
own post 5 with the paper PDF (CI artifact) for an engagement bump.
