from typing import Optional, Type, Any
from chromadb import Collection
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, Extra
from chromadb import Collection


# Anfrage, die Nutzende stellen
class SearchToolInput(BaseModel):
    query: Optional[str] = Field(
        None,
        description="nutzerfrage",
    )


# Werkzeug (Tool), um die Datenbank abzufragen
class SearchTool(BaseTool):
    name: str = "lehrbuch_abfragen"
    description: str = "Ermittle eine Antwort aus dem Lehrbuch auf Basis der Nutzerfrage."
    args_schema: Type[BaseModel] = SearchToolInput

    class Config:
        extra = Extra.allow

    def __init__(self, collection: Collection):
        """Initialisiere das Tool und lade die Vektordatenbank"""
        super().__init__()
        self.collection = collection

    def _run(self, query: Optional[str], **kwargs) -> Any:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=5,
            )
            print(results)
            return results
        except Exception as e:
            return f"Error querying items: {str(e)}"
