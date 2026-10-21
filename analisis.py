import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# 1. Configuración de Logging Corporativo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Motor IA] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

class AnalizadorSentimiento:
    def __init__(self, model_path: str = None) -> None:
        """Inicializa el motor con manejo de excepciones y detección de hardware."""
        if model_path is None:
            folder_name = os.getenv("MODEL_FOLDER_NAME", "modelo_beto")
            # resolve() asegura la ruta absoluta real previniendo errores en algunos OS
            base_dir = Path(__file__).resolve().parent
            self.model_path = base_dir / folder_name
        else:
            self.model_path = Path(model_path)

        # 2. Detección automática de aceleración por hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Iniciando analizador. Hardware asignado: {self.device.type.upper()}")

        # 3. Bloque Try-Except para carga segura de binarios
        try:
            logger.info(f"Cargando tensores desde: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            
            # Enviar el modelo a la GPU (si existe), si no, se queda en CPU
            self.model.to(self.device)
            
            self.mapa_etiquetas = {0: "NEGATIVO", 1: "NEUTRO", 2: "POSITIVO"}
            logger.info("Modelo cargado exitosamente y listo para inferencia.")
            
        except OSError as e:
            logger.error(f"FALLO CRÍTICO: Archivos de modelo no encontrados en {self.model_path}")
            raise RuntimeError(f"Error de inicialización: {e}")
        except Exception as e:
            logger.error(f"FALLO CRÍTICO: Error desconocido al cargar modelo: {e}")
            raise RuntimeError(f"Error de inicialización: {e}")

    def analizar_texto(self, texto: str) -> Dict[str, Any]:
        """
        Vectoriza el texto e infiere con validación estricta de tipos.
        """
        # 4. Validación de la calidad del dato entrante
        if not isinstance(texto, str) or not texto.strip():
            logger.warning("Se recibió un texto vacío o de tipo inválido. Abortando inferencia.")
            return {"error": "El dato de entrada debe ser un string con contenido."}

        try:
            # Vectorización
            inputs = self.tokenizer(
                texto, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Asegurar que los vectores estén en la misma memoria (RAM o VRAM) que el modelo
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inferencia matemática aislada
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Transformación Softmax a porcentajes
            probabilidades = F.softmax(outputs.logits, dim=-1)
            indice_ganador = torch.argmax(probabilidades, dim=-1).item()
            
            etiqueta = self.mapa_etiquetas[indice_ganador]
            certeza = probabilidades[0][indice_ganador].item() * 100
            
            return {
                "texto": texto,
                "clasificacion": etiqueta,
                "certeza_porcentaje": round(certeza, 2),
                "desglose": {
                    "NEGATIVO": round(probabilidades[0][0].item() * 100, 2),
                    "NEUTRO": round(probabilidades[0][1].item() * 100, 2),
                    "POSITIVO": round(probabilidades[0][2].item() * 100, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error durante la inferencia del texto: '{texto[:20]}...'. Detalles: {e}")
            return {"error": f"Fallo interno de procesamiento: {str(e)}"}