# graph/queries.py

from graph.db_connection import get_driver


# ── Write functions (take tx, used inside transactions) ──────────

def create_act(tx, act_id, name, year, jurisdiction, short_name):
    query = """
    MERGE (a:Act {act_id: $act_id})
    SET a.name = $name,
        a.year = $year,
        a.jurisdiction = $jurisdiction,
        a.short_name = $short_name
    RETURN a
    """
    return tx.run(query, act_id=act_id, name=name, year=year,
                  jurisdiction=jurisdiction, short_name=short_name).single()


def create_section(tx, section_id, act_id, number, title, text):
    query = """
    MATCH (a:Act {act_id: $act_id})
    MERGE (s:Section {section_id: $section_id})
    SET s.number = $number,
        s.title = $title,
        s.text = $text,
        s.act_id = $act_id
    MERGE (a)-[:HAS_SECTION]->(s)
    RETURN s
    """
    return tx.run(query, section_id=section_id, act_id=act_id,
                  number=number, title=title, text=text).single()


def create_concept(tx, concept_id, name, description="", aliases=None):
    query = """
    MERGE (c:Concept {concept_id: $concept_id})
    SET c.name = $name,
        c.description = $description,
        c.aliases = $aliases
    RETURN c
    """
    return tx.run(query, concept_id=concept_id, name=name,
                  description=description,
                  aliases=aliases or []).single()


def connect_concept_to_section(tx, concept_id, section_id):
    query = """
    MATCH (c:Concept {concept_id: $concept_id})
    MATCH (s:Section {section_id: $section_id})
    MERGE (s)-[:APPLIES_TO]->(c)
    RETURN c, s
    """
    return tx.run(query, concept_id=concept_id, section_id=section_id).single()


def create_section_relationship(tx, from_id, to_id, rel_type, properties=None):
    query = f"""
    MATCH (from:Section {{section_id: $from_id}})
    MATCH (to:Section {{section_id: $to_id}})
    MERGE (from)-[r:{rel_type}]->(to)
    SET r += $properties
    RETURN r
    """
    return tx.run(query, from_id=from_id, to_id=to_id,
                  properties=properties or {}).single()


# ── Read functions (open their own session) ──────────────────────

def get_section_by_id(section_id: str):
    driver = get_driver()
    query = """
    MATCH (s:Section {section_id: $section_id})
    RETURN s
    """
    with driver.session() as session:
        return session.run(query, section_id=section_id).single()


def get_sections_for_concept(concept_name: str):
    driver = get_driver()
    query = """
    MATCH (s:Section)-[:APPLIES_TO]->(c:Concept)
    WHERE toLower(c.name) CONTAINS toLower($concept_name)
       OR any(alias IN c.aliases WHERE toLower(alias) CONTAINS toLower($concept_name))
    RETURN c, s
    """
    with driver.session() as session:
        return list(session.run(query, concept_name=concept_name))


def get_neighbors(section_id: str):
    """
    Fetches all nodes directly connected to a section.
    Core function used by traversal.py for multi-hop expansion.
    """
    driver = get_driver()
    query = """
    MATCH (s:Section {section_id: $section_id})-[r]->(neighbor)
    RETURN s, type(r) as rel_type, neighbor
    """
    with driver.session() as session:
        return list(session.run(query, section_id=section_id))


def test_connection():
    driver = get_driver()
    with driver.session() as session:
        result = session.run("RETURN 'Neo4j connection OK' as message")
        print(result.single()["message"])


if __name__ == "__main__":
    test_connection()