SUPER_GRAPH_QUERY = """
MATCH (a)-[:CONNECTED_TO]->(b)
WITH DISTINCT a.label AS A, b.label AS B, COUNT(*) AS numConnections
MERGE (nodeA:SuperNode {label: A})
MERGE (nodeB:SuperNode {label: B})
MERGE (nodeA)-[r:SUPER_CONNECTED]->(nodeB)
SET r.count = numConnections;
"""

"""
Calculate the degree of the super graph
So we can know whether this graph is a fully connected/completed graph
"""

SUPER_GRAPH_DEGREE = """
MATCH (n:SuperNode)
WITH COLLECT(n) AS subgraphNodes
UNWIND subgraphNodes AS node

// Matching and counting incoming relationships
MATCH (node)<-[r_in]-()
WITH node, COUNT(r_in) AS inDegree

// Matching and counting outgoing relationships
MATCH (node)-[r_out]->()
RETURN node.label AS label, inDegree, COUNT(r_out) AS outDegree;

"""

IN_OUT_DEGREE_QUERY = """
MATCH (n)
OPTIONAL MATCH (n)-[out_r]->()
WITH n, COUNT(out_r) AS out_degree
MATCH (n)
OPTIONAL MATCH (n)<-[in_r]-()
RETURN n, n.node_id, COUNT(in_r) AS in_degree, out_degree, n.label as class_label
ORDER BY in_degree DESC, out_degree DESC;
"""

AVERAGE_IN_OUT_DEGREE_QUERY = """
MATCH (n)
OPTIONAL MATCH (n)-[out_r]->()
WITH n, COUNT(out_r) AS out_degree
MATCH (n)
OPTIONAL MATCH (n)<-[in_r]-()
WITH n.label AS class, COUNT(in_r) AS in_degree, out_degree
WITH class, AVG(in_degree) AS avg_in_degree, AVG(out_degree) AS avg_out_degree
RETURN class, avg_in_degree, avg_out_degree
ORDER BY avg_in_degree DESC, avg_out_degree DESC;
"""
