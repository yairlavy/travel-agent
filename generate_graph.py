"""
Generates graph.png from a hand-crafted Mermaid diagram.

Solid arrows  -->   fixed edges (always taken)
Dashed arrows -.->  conditional edges (router decides at runtime)

Renders via mermaid.ink (no local install required).
"""

import base64
import httpx

MERMAID = """flowchart TD
    S([START]) --> EM

    EM[extract_metadata]

    EM -. recall query .-> RC
    EM -. preference stated .-> UP
    EM --> V

    RC[recall]
    RC --> E1([END])

    UP[update_preferences]
    UP --> V

    V[validator\nGroq AI + regex fallback]

    V -. blocked .-> E2([END])
    V -. research query .-> RE
    V --> A

    RE[researcher\ndirect DB - no LLM]
    RE --> E3([END])

    A[agent - Marco\nGemini or Groq LLM]

    A -. has tool calls .-> T
    A -. count over 8 or loop .-> CB
    A -. admin + full plan .-> RV
    A --> SM

    T[tools\nfetch flights - hotels\nactivities - visa - cost]
    T --> A

    CB[circuit_breaker\nloop guard]
    CB --> E4([END])

    RV[reviewer\nadmin sessions only]
    RV --> SM

    SM[summarizer\ncompacts history]
    SM --> E5([END])

    style S  fill:#263238,color:#fff,stroke:#263238
    style E1 fill:#263238,color:#fff,stroke:#263238
    style E2 fill:#263238,color:#fff,stroke:#263238
    style E3 fill:#263238,color:#fff,stroke:#263238
    style E4 fill:#263238,color:#fff,stroke:#263238
    style E5 fill:#263238,color:#fff,stroke:#263238
    style EM fill:#0277bd,color:#fff,stroke:#01579b
    style UP fill:#00695c,color:#fff,stroke:#004d40
    style RC fill:#00695c,color:#fff,stroke:#004d40
    style V  fill:#e65100,color:#fff,stroke:#bf360c
    style RE fill:#1565c0,color:#fff,stroke:#0d47a1
    style A  fill:#1565c0,color:#fff,stroke:#0d47a1
    style T  fill:#1565c0,color:#fff,stroke:#0d47a1
    style CB fill:#b71c1c,color:#fff,stroke:#7f0000
    style RV fill:#f9a825,color:#000,stroke:#f57f17
    style SM fill:#6a1b9a,color:#fff,stroke:#4a148c"""


def render(mermaid_str: str, output: str = "graph.png") -> None:
    encoded = base64.urlsafe_b64encode(mermaid_str.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=fff"
    print("Rendering via mermaid.ink ...")
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    with open(output, "wb") as f:
        f.write(resp.content)
    print(f"Graph image saved as {output}")


if __name__ == "__main__":
    render(MERMAID)
