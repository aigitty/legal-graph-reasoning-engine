# Create root folder
New-Item -ItemType Directory -Path "legal-graph-rag" -Force

# Move into root folder
Set-Location "legal-graph-rag"

# Create folders
$folders = @(
    "data",
    "data/acts",
    "data/ontology",
    "data/processed",
    "ingest",
    "graph",
    "agents",
    "guardrails",
    "guardrails/config"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path $folder -Force
}

# Create files
$files = @(
    ".env",
    "requirements.txt",
    "README.md",
    "main.py",

    "data/ontology/concept_map.json",

    "data/processed/sections.jsonl",
    "data/processed/relationships.jsonl",

    "ingest/pdf_parser.py",
    "ingest/ontology_loader.py",
    "ingest/graph_builder.py",

    "graph/schema.py",
    "graph/queries.py",
    "graph/traversal.py",

    "agents/state.py",
    "agents/nodes.py",
    "agents/graph_agent.py",

    "guardrails/input_rail.py",
    "guardrails/output_rail.py",
    "guardrails/config/legal_rails.co"
)

foreach ($file in $files) {
    New-Item -ItemType File -Path $file -Force
}

Write-Host ""
Write-Host "✅ Legal Graph RAG project structure created successfully!"