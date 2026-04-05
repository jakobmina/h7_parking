import os
import json
import logging
import random
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, SafetySetting, Part
except ImportError:
    vertexai = None

try:
    from transformers import AutoTokenizer, pipeline
    from optimum.onnxruntime import ORTModelForCausalLM
    LOCAL_ONNX_SUPPORT = True
except ImportError:
    LOCAL_ONNX_SUPPORT = False

logger = logging.getLogger(__name__)

class GemmaMetriplexOracle:
    """
    Oráculo de Información usando Gemma 4.
    Traduce lenguaje natural (infracciones) a las densidades iniciales (rho y v)
    requeridas para el Clasificador QML Metripléctico.
    """
    def __init__(self):
        self.project_id = os.getenv("VERTEX_PROJECT_ID")
        self.region = os.getenv("VERTEX_REGION", "us-central1")
        self.model_name = os.getenv("GEMMA_BASE_MODEL", "gemma-4-26b-a4b-it")
        self.use_mock = False
        self.use_offline_onnx = False
        self.local_onnx_pipeline = None

        if not self.project_id or vertexai is None:
            logger.warning("Credenciales de Vertex AI no encontradas o vertexai no instalado. Activando fallback Offline (ONNX).")
            self._init_offline_onnx()
        else:
            try:
                vertexai.init(project=self.project_id, location=self.region)
                publisher_model_name = f"publishers/google/models/gemma4@{self.model_name.lower()}"
                self.model = GenerativeModel(publisher_model_name)
            except Exception as e:
                logger.error(f"Fallo al inicializar VertexAI: {e}. Activando fallback Offline (ONNX).")
                self._init_offline_onnx()

    def _init_offline_onnx(self):
        """Inicializa el modelo Gemma E2B/E4B de forma local y offline usando ONNX Runtime."""
        if LOCAL_ONNX_SUPPORT:
            try:
                logger.info("Inicializando ONNX Runtime para Gemma...")
                # Se puede usar gemma-2b-it u otro como default para Edge
                edge_model_id = os.getenv("GEMMA_EDGE_MODEL", "google/gemma-2b-it") 
                
                # Cargamos tokenizer y el ORTModel (Optimizado para ONNX)
                tokenizer = AutoTokenizer.from_pretrained(edge_model_id)
                # NOTA: download_kwargs o uso de modelos locales exportados
                onnx_model = ORTModelForCausalLM.from_pretrained(edge_model_id, export=True) 
                
                self.local_onnx_pipeline = pipeline(
                    "text-generation",
                    model=onnx_model,
                    tokenizer=tokenizer,
                    max_new_tokens=100
                )
                self.use_offline_onnx = True
                logger.warning("ONNX Edge Model cargado correctamente en memoria local.")
            except Exception as e:
                logger.error(f"Fallo carga profunda de ONNX (¿pesos muy grandes o sin conexión inicial?): {e}. Usando mock temporal.")
                self.use_mock = True
        else:
            logger.error("ONNX/Transformers no instalados o no configurados. Usando mock.")
            self.use_mock = True

    def get_initial_phase_state(self, context: str, domain_prompt: str = "Analiza el siguiente contexto:") -> dict:
        """
        Consulta a Gemma 4 para obtener las variables físicas:
        rho: Densidad de la infracción (magnitud/gravedad percibida, 0 a 1)
        v: Velocidad/Intencionalidad (flujo de comportamiento, -1 a 1)
        """
        if self.use_mock:
            # Hash del contexto para mantener consistencia en la simulación
            h = sum(ord(c) for c in context)
            random.seed(h)
            rho = round(random.uniform(0.1, 1.0), 4)
            v = round(random.uniform(-1.0, 1.0), 4)
            return {"rho": rho, "v": v}

        prompt = f"""
        Eres un Oráculo Físico Metripléctico.
        {domain_prompt}
        "{context}"
        
        Evalúa y extrae exclusivamente dos valores en formato JSON estricto:
        1. "rho": Densidad de impacto o peso real en el sistema (de 0.1 a 1.0).
        2. "v": Intencionalidad o dirección del flujo asociado (-1.0 a 1.0, donde negativo es perjudicial/destructivo y positivo es ordenado/constructivo).
        
        Ejemplo: {{"rho": 0.8, "v": -0.5}}
        Solo responde con el JSON.
        """

        try:
            if self.use_offline_onnx and self.local_onnx_pipeline:
                # Inferencia puramente OFF-LINE, por CPU / Edge via ONNX
                response = self.local_onnx_pipeline(prompt)
                content = response[0]["generated_text"].replace(prompt, "").strip()
            else:
                # Inferencia CLOUD (Vertex AI Gemma 4)
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 100}
                )
                content = response.text.strip()

            # Limpieza básica por si el modelo envuelve en markdown
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            data = json.loads(content)
            return {"rho": float(data.get("rho", 0.5)), "v": float(data.get("v", 0.0))}
        except Exception as e:
            logger.error(f"Error consultando al Oráculo (ONNX/Cloud falló): {e}")
            return {"rho": 0.5, "v": 0.0}
