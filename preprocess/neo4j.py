from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        with self.__driver.session(database=db) as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def get_neighbor(self, node_cui):
        query = """
                MATCH (n)-[]->(m) 
                WHERE n.cui = $node_cui
                RETURN m.cui AS neighbor_id
                """
        parameters = {'node_cui': node_cui}
        result = self.query(query, parameters)
        return [record["neighbor_id"] for record in result]
    
    def check_connect(self, node_a, node_b):
        query = """
                MATCH (a),(b)
                WHERE a.cui = $cui1 AND b.cui = $cui2
                RETURN EXISTS((a)-[]-(b)) AS hasRelationship
                """
        parameters = {'cui1': node_a, 'cui2': node_b}
        result = self.query(query, parameters)
        return [record["hasRelationship"] for record in result][0]
    
    def get_edge(self, node_a, node_b):
        query = """
                MATCH (a)-[r]->(b) 
                WHERE a.cui = $cui1 AND b.cui = $cui2
                RETURN TYPE(r) AS relationshipType
                """
        parameters = {'cui1': node_a, 'cui2': node_b}
        result = self.query(query, parameters)
        return [record["relationshipType"] for record in result][0]