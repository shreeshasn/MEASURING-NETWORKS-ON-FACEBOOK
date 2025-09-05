import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def analyze_network(n=50, p=0.08, seed=42):
    """
    Generate a synthetic Facebook-like network and analyze it.
    Parameters:
        n (int): number of nodes
        p (float): edge probability
        seed (int): random seed for reproducibility
    """

    # Generate Erdős–Rényi graph
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)

    # ---- Network Visualizations ----
    # Raw layout
    plt.figure(figsize=(5, 4))
    nx.draw(G, node_size=50, with_labels=False)
    plt.title("Fig. 1. Raw Facebook-like friendship network")
    plt.show()

    # Force-directed layout
    plt.figure(figsize=(5, 4))
    nx.draw_spring(G, node_size=50, with_labels=False)
    plt.title("Fig. 2. Force-directed layout")
    plt.show()

    # ---- Centrality Measures ----
    deg_cent = nx.degree_centrality(G)
    close_cent = nx.closeness_centrality(G)
    betw_cent = nx.betweenness_centrality(G)
    eig_cent = nx.eigenvector_centrality_numpy(G)

    # Degree distribution
    degrees = [deg for _, deg in G.degree()]
    plt.figure(figsize=(5, 4))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2))
    plt.title("Fig. 3. Degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

    # Helper: bar chart for top-k users
    def plot_topk(metric_dict, title, fig_number, k=10):
        topk = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        nodes, values = zip(*topk)
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(nodes)), values, tick_label=nodes)
        plt.title(f"Fig. {fig_number}. {title}")
        plt.xlabel("Node")
        plt.ylabel("Score")
        plt.show()
        return topk

    top_degree = plot_topk(deg_cent, "Top-10 users by degree centrality", 4)
    top_close = plot_topk(close_cent, "Top-10 users by closeness centrality", 5)
    top_betw = plot_topk(betw_cent, "Top-10 users by betweenness centrality", 6)
    top_eig = plot_topk(eig_cent, "Top-10 users by eigenvector centrality", 7)

    # ---- Graph Energy ----
    A = nx.to_numpy_array(G)
    eigenvals = np.linalg.eigvals(A)
    graph_energy = np.sum(np.abs(eigenvals))

    plt.figure(figsize=(6, 4))
    plt.plot(sorted(np.abs(eigenvals), reverse=True), marker="o")
    plt.title("Fig. 8. Eigenvalue spectrum (Graph Energy)")
    plt.xlabel("Index")
    plt.ylabel("Abs(Eigenvalue)")
    plt.show()

    # ---- Print summary ----
    print("\nTop-5 by Degree:", top_degree[:5])
    print("Top-5 by Closeness:", top_close[:5])
    print("Top-5 by Betweenness:", top_betw[:5])
    print("Top-5 by Eigenvector:", top_eig[:5])
    print("Graph Energy:", graph_energy)

    return {
        "degree": top_degree[:5],
        "closeness": top_close[:5],
        "betweenness": top_betw[:5],
        "eigenvector": top_eig[:5],
        "energy": graph_energy,
    }

# Run with default parameters (50 nodes, p=0.08)
if __name__ == "__main__":
    results = analyze_network()
