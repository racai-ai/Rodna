"""This module implements the Chu-Liu/Edmonds' Algorithm for Max Spanning Tree in a Di-graph.
Implementation taken from https://wendy-xiao.github.io/posts/2020-07-10-chuliuemdond_algorithm/
and adapted for Rodna."""

from typing import List, Union


def _reverse_graph(graph: dict) -> dict:
    """Return the reversed graph where g[dst][src]=graph[src][dst]"""

    g = {}

    for src in graph.keys():
        for dst in graph[src].keys():
            if dst not in g.keys():
                g[dst] = {}
            # end if

            g[dst][src] = graph[src][dst]
        # end for
    # end for

    return g


def _build_max(rg: dict, root: int) -> dict:
    """Find the max in-edge for every node except for the root."""

    mg = {}

    for dst in rg.keys():
        if dst == root:
            continue
        # end if

        max_ind = -100
        max_value = -100

        for src in rg[dst].keys():
            if rg[dst][src] >= max_value:
                max_ind = src
                max_value = rg[dst][src]
            # end if
        # end for

        mg[dst] = {max_ind: max_value}
    # end for

    return mg


def _find_circle(mg: dict) -> Union[List[int], None]:
    """Return the firse circle if found, otherwise return None."""

    for start in mg.keys():
        visited = []
        stack = [start]
        
        while stack:
            n = stack.pop()
            
            if n in visited:
                C = []

                while n not in C:
                    C.append(n)
                    n = list(mg[n].keys())[0]
                # end while

                return C
            # end if

            visited.append(n)

            if n in mg.keys():
                stack.extend(list(mg[n].keys()))
            # end if
        # end while
    # end for

    return None


def chu_liu_edmonds(graph: dict, root: int) -> dict:
    """ `graph`: dict of dict of weights (named G below, in the comments)
            graph[i][j] = w means the edge from node i to node j has weight w.
            Assume the graph is connected and there is at least one spanning tree existing in graph.
        `root`: the root node, has outgoing edges only.
    """

    # reversed graph rg[dst][src] = G[src][dst]
    rg = _reverse_graph(graph)
    # root only has out edge
    rg[root] = {}
    # the maximum edge for each node other than root
    mg = _build_max(rg, root)
    # check if mg is a tree (contains a circle) named C
    cl = _find_circle(mg)

    # if there is no circle, it means mg is what we want
    if not cl:
        return _reverse_graph(mg)
    # end if

    # Now consider the nodes in the circle C as one new node vc
    all_nodes = graph.keys()
    vc = max(all_nodes) + 1

    # The new graph G_prime with V_prime=V\C+{vc}
    g_prime = {}
    vc_in_idx = {}
    vc_out_idx = {}
    
    # Now add the edges to g_prime
    for u in all_nodes:
        for v in graph[u].keys():
            # First case: if the source is not in the circle, and the dest is in the circle, i.e. in-edges for C
            # Then we only keep one edge from each node that is not in C to the new node vc with the largest difference (G[u][v]-list(mg[v].values())[0])
            # To specify, for each node u in V\C, there is an edge between u and vc if and only if there is an edge between u and any node v in C,
            # And the weight of edge u->vc = max_{v in C} (G[u][v] - mg[v].values) The second term represents the weight of max in-edge of v.
            # Then we record that the edge u->vc is originally the edge u->v with v=argmax_{v in C} (G[u][v] - mg[v].values)

            if (u not in cl) and (v in cl):
                if u not in g_prime.keys():
                    g_prime[u] = {}
                # end if

                w = graph[u][v] - list(mg[v].values())[0]
                
                if (vc not in g_prime[u]) or (vc in g_prime[u] and w > g_prime[u][vc]):
                    g_prime[u][vc] = w
                    vc_in_idx[u] = v
                # end if
            # Second case: if the source is in the circle, but the dest is not in the circle, i.e out-edge for C
            # Then we only keep one edge from the new node vc to each node that is not in C
            # To specify, for each node v in V\C, there is an edge between vc and v iff there is an edge between any edge u in C and v.
            # And the weight of edge vc->v = max_{u in C} G[u][v]
            # Then we record that the edge vc->v originally the edge u->v with u=argmax_{u in C} G[u][v]
            elif (u in cl) and (v not in cl):
                if vc not in g_prime.keys():
                    g_prime[vc] = {}
                # end if
                
                w = graph[u][v]
                
                if (v not in g_prime[vc]) or (v in g_prime[vc] and w > g_prime[vc][v]):
                    g_prime[vc][v] = w
                    vc_out_idx[v] = u
                # end if
            # Third case: if the source and dest are all not in the circle, then just add the edge to the new graph.
            elif (u not in cl) and (v not in cl):
                if u not in g_prime.keys():
                    g_prime[u] = {}
                # end if

                g_prime[u][v] = graph[u][v]
            # end if
        # end for
    # end for

    # Recursively run the algorihtm on the new graph G_prime
    # The result A should be a tree with nodes V\C+vc, then we just need to break the circle C and plug the subtree into A
    # To break the circle, we need to use the in-edge of vc, say u->vc to replace the original selected edge u->v,
    # where v was the original edge we recorded in the first case above.
    # Then if vc has out-edges, we also need to replace them with the original edges, recorded in the second case above.
    # This one used to be called A
    ag = chu_liu_edmonds(g_prime, root)
    all_nodes_ag = list(ag.keys())
    orig_in = -1

    for src in all_nodes_ag:
        # The number of out-edges varies, could be 0 or any number <=|V\C|
        if src == vc:
            for node_in in ag[src].keys():
                orig_out = vc_out_idx[node_in]

                if orig_out not in ag.keys():
                    ag[orig_out] = {}
                # end if

                ag[orig_out][node_in] = graph[orig_out][node_in]
            # end for
        elif vc in ag[src]:
            # There must be only one in-edge to vc.
            orig_in = vc_in_idx[src]
            ag[src][orig_in] = graph[src][orig_in]
            del ag[src][vc]
        # end if
    # end for

    if vc in ag:
        del ag[vc]
    # end if

    # Now add the edges from the circle to the result.
    # Remember not to include the one with new in-edge
    for node in cl:
        if node != orig_in:
            src = list(mg[node].keys())[0]

            if src not in ag.keys():
                ag[src] = {}
            # end if

            ag[src][node] = mg[node][src]
        # end if
    # end for

    return ag
