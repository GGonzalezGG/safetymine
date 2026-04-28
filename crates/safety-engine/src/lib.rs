use common::{DetectionResult, MinaContext, EppItem};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize)]
struct Database {
    minas: Vec<MinaContext>,
}

pub struct SafetyEvaluator {
    pub contexto_actual: Option<MinaContext>,
}

impl SafetyEvaluator {
    /// Inicializa el evaluador cargando la "DB" y seleccionando una mina por ID
    pub fn new(db_path: &str, mina_id: &str) -> Self {
        let db_content = fs::read_to_string(db_path)
            .expect("No se pudo leer la base de datos JSON");
        
        let db: Database = serde_json::from_str(&db_content)
            .expect("Error al parsear el JSON de la base de datos");

        // Buscamos la mina específica en nuestra "DB"
        let contexto = db.minas.into_iter()
            .find(|m| m.nombre_mina.contains(mina_id)); // O puedes usar un campo ID

        Self {
            contexto_actual: contexto,
        }
    }

    /// Compara los items detectados por la visión con los obligatorios de la mina
    pub fn verificar_faltantes(&self, deteccion: &DetectionResult) -> Vec<EppItem> {
        if let Some(ref ctx) = self.contexto_actual {
            let labels_detectados: Vec<EppItem> = deteccion.detecciones.iter()
                .map(|d| d.label.clone())
                .collect();

            ctx.epp_obligatorio.iter()
                .filter(|item| !labels_detectados.contains(item))
                .cloned()
                .collect()
        } else {
            vec![]
        }
    }

    /// Lee el contenido del protocolo (Markdown) para pasárselo al LLM
    pub fn obtener_texto_protocolo(&self) -> String {
        if let Some(ref ctx) = self.contexto_actual {
            fs::read_to_string(&ctx.path_protocolo)
                .unwrap_or_else(|_| format!("Error: No se encontró el archivo en {}", ctx.path_protocolo))
        } else {
            "No hay contexto de mina cargado.".to_string()
        }
    }
}