"""
Sistema RAG simple con BM25 sobre datos de rendimiento estudiantil.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re

class RendimientoRAG:
    """RAG básico para búsqueda en datos de rendimiento estudiantil."""
    
    def __init__(self, csv_path: str | Path = "datasets/csvClear/Rendimiento_all_final.csv", max_rows: int = 50_000):
        """
        Carga muestra del CSV y construye índice BM25.
        
        Args:
            csv_path: Ruta al CSV de rendimiento
            max_rows: Máximo de filas a cargar (para performance)
        """
        self.documents = []
        self.metadata = []
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"⚠️ CSV no encontrado: {csv_path}. RAG deshabilitado.")
            self.bm25 = None
            return
        
        try:
            # Cargar muestra
            df = pd.read_csv(csv_path, nrows=max_rows)
            
            # Filtrar filas con datos completos
            required_cols = ["AGNO", "PROM_GRAL", "ASISTENCIA", "SIT_FIN"]
            available = [c for c in required_cols if c in df.columns]
            df = df[available].dropna(subset=available[:2])  # Al menos año y promedio
            
            # Construir documentos textuales (cada fila = un documento)
            for _, row in df.iterrows():
                doc_text = self._row_to_text(row)
                self.documents.append(doc_text)
                self.metadata.append(row.to_dict())
            
            # Tokenizar y crear índice BM25
            tokenized = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
            
            print(f"✅ RAG inicializado con {len(self.documents):,} registros de rendimiento")
        
        except Exception as e:
            print(f"⚠️ Error inicializando RAG: {e}")
            self.bm25 = None
    
    def _row_to_text(self, row: pd.Series) -> str:
        """Convierte una fila del CSV en texto descriptivo."""
        parts = []
        
        if "AGNO" in row and pd.notna(row["AGNO"]):
            parts.append(f"año {int(row['AGNO'])}")
        
        if "PROM_GRAL" in row and pd.notna(row["PROM_GRAL"]):
            nota = float(row["PROM_GRAL"])
            nivel = "excelente" if nota >= 6.0 else "bueno" if nota >= 5.0 else "suficiente" if nota >= 4.0 else "bajo"
            parts.append(f"promedio {nota:.1f} ({nivel})")
        
        if "ASISTENCIA" in row and pd.notna(row["ASISTENCIA"]):
            asist = float(row["ASISTENCIA"])
            nivel_asist = "alta" if asist >= 85 else "media" if asist >= 70 else "baja"
            parts.append(f"asistencia {asist:.0f}% ({nivel_asist})")
        
        if "SIT_FIN" in row and pd.notna(row["SIT_FIN"]):
            sit = str(row["SIT_FIN"]).strip().upper()
            estado = "aprobado" if sit in ("P", "APROBADO") else "reprobado" if sit in ("R", "REPROBADO") else "retirado"
            parts.append(f"situación {estado}")
        
        if "NOM_RBD" in row and pd.notna(row["NOM_RBD"]):
            parts.append(f"establecimiento {row['NOM_RBD']}")
        
        return " ".join(parts)
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokeniza texto (lowercase, sin puntuación)."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Busca registros relevantes.
        
        Args:
            query: Pregunta del usuario
            top_k: Cantidad de resultados
        
        Returns:
            Lista de dicts con 'text', 'metadata' y 'score'
        """
        if self.bm25 is None or not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Top K índices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(scores[idx])
                })
        
        return results
    
    def format_context(self, results: list[dict]) -> str:
        """Formatea resultados de búsqueda para el prompt del LLM."""
        if not results:
            return "No se encontraron datos históricos relevantes."
        
        lines = ["Datos históricos relevantes:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['text']} (score: {r['score']:.1f})")
        
        return "\n".join(lines)