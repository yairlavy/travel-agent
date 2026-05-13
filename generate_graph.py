"""
Generates graph.png from a hand-crafted Mermaid diagram.

Layout:
  Spine (center): START -> extract_metadata -> validator -> agent -> summarizer -> END
  Side nodes branch off the spine. All paths lead to the single END node,
  with a label on each edge explaining why that path terminated.

Solid arrows  -->    fixed edges
Dashed arrows -.->   conditional edges (router decides at runtime)
"""

import base64
import httpx

MERMAID = """%%{init: {'flowchart': {'curve': 'stepBefore', 'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

    S([START]) --> EM

    EM[extract_metadata]
    EM --> V

    V[validator]
    V -->|approved| A

    A[agent - Marco]
    A --> SM

    SM[summarizer]
    SM -->|turn complete| E([END])

    EM -.->|preference stated| UP
    UP[update_preferences]
    UP --> V

    EM -.->|recall query| RC
    RC[recall]
    RC -->|memory answered| E

    V -.->|request blocked| E
    V -.->|research query| RE
    RE[researcher]
    RE -->|data lookup complete| E

    A -.->|has tool calls| T
    T[tools]
    T --> A

    A -.->|count over 8 or loop| CB
    CB[circuit_breaker]
    CB -->|loop stopped| E

    A -.->|admin and full plan| RV
    RV[reviewer]
    RV --> SM

    style S  fill:#1a1a2e,color:#fff,stroke:#1a1a2e
    style E  fill:#1a1a2e,color:#fff,stroke:#1a1a2e

    style EM fill:#0277bd,color:#fff,stroke:#01579b
    style V  fill:#e65100,color:#fff,stroke:#bf360c
    style A  fill:#1565c0,color:#fff,stroke:#0d47a1
    style SM fill:#6a1b9a,color:#fff,stroke:#4a148c

    style UP fill:#2e7d32,color:#fff,stroke:#1b5e20
    style RC fill:#2e7d32,color:#fff,stroke:#1b5e20
    style RE fill:#00838f,color:#fff,stroke:#006064
    style T  fill:#1565c0,color:#fff,stroke:#0d47a1
    style CB fill:#b71c1c,color:#fff,stroke:#7f0000
    style RV fill:#f57f17,color:#000,stroke:#e65100"""


def render(mermaid_str: str, output: str = "graph.png") -> None:
    encoded = base64.urlsafe_b64encode(mermaid_str.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=transparent"
    print("Rendering via mermaid.ink ...")
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    with open(output, "wb") as f:
        f.write(resp.content)
    print(f"Saved as {output}")


if __name__ == "__main__":
    render(MERMAID)
