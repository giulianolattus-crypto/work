import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_causal_networks_grid(network_data_list, positions_list=None, titles=None,row_labels=['SON', 'DJF'], figsize=(32, 16)):
    """
    Plots 8 causal directed networks in a 2x4 grid layout.

    Parameters:
    - network_data_list: list of 8 dicts. Each dict contains edge tuples (source, target) as keys,
                         and a 'mean' value as the attribute, e.g. {("ENSO", "SPV"): 0.3, ...} (not really mean value anymore, but should still work?)
    - positions_list: optional list of 8 dicts with node positions, e.g. {"ENSO": (0,2), ...}
    - titles: optional list of 8 subplot titles
    - figsize: size of the entire figure
    """
    rows=int(len(network_data_list)/4)
    cols=int(len(network_data_list)/2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for idx, edge_dict in enumerate(network_data_list):
        ax = axes[idx]
        G = nx.DiGraph()
        G.add_edges_from(edge_dict.keys())

        # Assign mean as edge attribute
        for (u, v), mean_val in edge_dict.items():
            G[u][v]['mean'] = mean_val

        # Use provided or default position
        if positions_list and positions_list[idx]:
            pos_raw = positions_list[idx]
            pos = {}

            for node, xy in pos_raw.items():
                pos[node] = xy

            # map dataset nodes ("T_Andes") -> layout nodes ("T")
            pos_fixed = {}
            for node in G.nodes():
                base = node.split("_")[0]  # T_Andes -> T
                if base in pos:
                    pos_fixed[node] = pos[base]
                else:
                    pos_fixed[node] = pos_raw.get(node, (0, 0))

            pos = pos_fixed
        else:
            pos = nx.spring_layout(G, seed=42)

        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=4500, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=26, font_weight='bold', ax=ax)
        
        edge_colors = []
        edge_widths = []
        for u, v, d in G.edges(data=True):
            weight = d['mean']
            edge_widths.append(1 + 5 * abs(weight))  # thickness based on magnitude

            # Color only if the edge points into T or Precip
            if v == 'T_Andes' or v=='T_LP':
                color = 'red' if weight >= 0 else 'blue'
            elif v == 'Precip_Andes' or v=='Precip_LP':
                color = 'green' if weight >= 0 else '#b8860b'  # ochre
            else:
                color = 'gray'  # default color for other edges
            edge_colors.append(color)
            
        # Draw arrows with curvature
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=30,
            min_source_margin=40,
            min_target_margin=40,
            ax=ax
        )

        # Add edge labels (means)
        edge_labels = {(u, v): f"{d['mean']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=26, label_pos=0.5, ax=ax)

        # Subplot title
        if titles:
            ax=axes[int(idx/2)]
            ax.set_title(titles[int(idx/2)], fontsize=44, fontweight='bold')
        ax=axes[idx]
        ax.axis('off')
   
     # Add row labels (left side of first column)
    if row_labels and len(row_labels) == 2:
        for row_idx, label in enumerate(row_labels):
            ax = axes[row_idx * 4]  # First column in each row
            ax.annotate(
                label,
                xy=(-0.2, 0.5),
                xycoords='axes fraction',
                textcoords='offset points',
                ha='right',
                va='center',
                fontsize=44,
                fontweight='bold'
            )
    fig.suptitle("Causal Network with Whole Period Regression Coefficient Values", fontsize=60,fontweight='heavy', y=1.02)
    plt.tight_layout()
    fig.savefig('DAG_with_reg_vals.pdf', bbox_inches='tight')
    plt.show()



