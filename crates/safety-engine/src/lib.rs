use common::{DetectionResult, MinaContext, EppItem};
use std::fs;

pub struct SafetyEvaluator {
    pub contexto: MinaContext,
}

impl SafetyEvaluator {
    pub fn new(config_path: &str) -> Self {
        // Aquí cargarías el JSON de configuración
        // Por ahora, devolvemos uno manual para el PoC
        Self {
            contexto: MinaContext {
                nombre_mina: "Mina Norte".to_string(),
                riesgos_activos: vec!["Gases Tóxicos".to_string()],
                epp_obligatorio: vec![EppItem::Casco, EppItem::Respirador],
                path_protocolo: "protocols/norte.md".to_string(),
            },
        }
    }

    /// Compara lo detectado vs lo obligatorio y retorna lo que falta
    pub fn verificar_faltantes(&self, deteccion: &DetectionResult) -> Vec<EppItem> {
        self.contexto.epp_obligatorio.iter()
            .filter(|item| !deteccion.items_detectados.contains(item))
            .cloned()
            .collect()
    }

    /// Lee el contenido del archivo Markdown del protocolo
    pub fn obtener_texto_protocolo(&self) -> String {
        fs::read_to_string(&self.contexto.path_protocolo)
            .unwrap_or_else(|_| "Protocolo estándar de seguridad".to_string())
    }
}