from src.graph.workflow import graph

# ASCII preview in terminal
print(graph.get_graph().draw_ascii())

# Save PNG
png = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png)

print("Graph image saved as graph.png")